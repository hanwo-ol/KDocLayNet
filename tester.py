import os
import json
import random
from PIL import Image
import numpy as np
import cv2
import time
import datetime # 결과 폴더 이름 생성용
import shutil # 폴더 관리용

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics # 평가 지표용

import matplotlib.pyplot as plt # 시각화용
from tqdm import tqdm

# --- 1. 설정 (Configuration) ---
DATA_DIR = "DocLayNet_core" # <<< DocLayNet 데이터셋 경로
MODEL_PATH = os.path.join("saved_models", "best_model.pth") 

# 테스트 데이터 정보
ANNOTATION_FILE_TEST = os.path.join(DATA_DIR, "COCO", "test.json") # <<< 테스트셋 어노테이션 경로
IMAGE_DIR_TEST = os.path.join(DATA_DIR, "PNG") # 학습/검증 시 사용한 이미지 폴더와 동일

# 결과 저장 폴더 설정
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"{current_time}_result"
METRICS_FILE = os.path.join(RESULT_DIR, "metrics.txt")
VISUALIZATION_DIR = os.path.join(RESULT_DIR, "visualizations") # 시각화 결과 저장 폴더

# 클래스 정보 (학습 시와 동일)
CLASS_NAMES = [
    'BACKGROUND', 'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# 하이퍼파라미터 (모델 구조 및 입력 크기는 학습 시와 동일해야 함)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512 # 학습 시 사용한 이미지 크기
BATCH_SIZE = 8 # 평가 시 배치 크기 (메모리에 맞게 조절)
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0
NUM_WORKERS = min(NUM_WORKERS, 4) # 예시: 최대 4개로 제한

# 시각화 설정
SAVE_VISUALIZATIONS = True # True로 설정하면 예측 결과 시각화 이미지 저장
MAX_VISUALIZATIONS = 20    # 저장할 최대 시각화 이미지 수 (None이면 모두 저장)
VIS_ALPHA = 0.6 # 시각화 시 마스크 투명도

# 시각화용 색상맵 (클래스별 랜덤 색상, 배경은 검정)
random.seed(42)
COLOR_MAP = {0: (0, 0, 0)} # 배경: 검정
for i in range(1, NUM_CLASSES):
    COLOR_MAP[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# --- 2. 데이터셋 클래스 (train.py와 동일) ---
# (DocLayNetDataset 클래스 정의는 train.py와 동일하게 사용)
class DocLayNetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, class_map, transforms=None, return_original=False, return_filename=False):
        self.image_dir = image_dir
        print(f"Loading annotations from: {annotation_file}")
        try:
            self.coco = COCO(annotation_file)
        except Exception as e:
            print(f"Error loading COCO file {annotation_file}: {e}")
            raise
        print(f"Annotations loaded successfully.")

        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.class_map = class_map
        self.transforms = transforms
        self.return_original = return_original
        self.return_filename = return_filename

        self.cat_ids = self.coco.getCatIds(catNms=[name for name in class_map.keys() if name != 'BACKGROUND'])
        self.cat_id_to_label = {cat_id: self.class_map[self.coco.loadCats(cat_id)[0]['name']]
                                for cat_id in self.cat_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        filename = img_info['file_name']

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # 오류 발생 시 대체 처리 (예: None 반환 후 collate_fn에서 처리)
            return None # 오류 시 None 반환

        original_image = image.copy() if self.return_original else None
        original_h, original_w = image.shape[:2]

        mask = np.zeros((original_h, original_w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann['category_id']
            class_label = self.cat_id_to_label.get(category_id)

            if class_label is not None and 'segmentation' in ann:
                if isinstance(ann['segmentation'], list): # Polygon
                     if len(ann['segmentation']) > 0:
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:
                                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                                cv2.fillPoly(mask, [poly], color=class_label)
                # RLE 처리 (필요시 추가)


        # 변환 적용
        transformed_image = None
        transformed_mask = None
        if self.transforms:
            try:
                augmented = self.transforms(image=image, mask=mask)
                transformed_image = augmented['image']
                transformed_mask = augmented['mask']
            except Exception as e:
                 print(f"Error during augmentation for image {img_info['file_name']}: {e}")
                 # 변환 오류 시 None 반환
                 return None

        # 변환된 마스크가 None이 아니면 long 타입으로 변경
        if transformed_mask is not None:
            transformed_mask = transformed_mask.long()

        # 반환할 값들을 튜플로 구성
        results = (transformed_image, transformed_mask)
        if self.return_original:
            results += (original_image,)
        if self.return_filename:
            results += (filename,)

        return results

# 데이터 로더에서 None 값을 걸러내기 위한 collate_fn
def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # 만약 필터링 후 배치가 비어있다면
        return None # 혹은 다른 처리 (예: 빈 텐서 반환)
    return torch.utils.data.dataloader.default_collate(batch)


# --- 3. 데이터 변환 (테스트 시에는 Augmentation 없이 Resize, Normalize만 적용) ---
test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 학습 시와 동일한 값
    ToTensorV2(),
])

# --- 4. 모델 로드 ---
print("=" * 30)
print("      Model Testing Start       ")
print("=" * 30)
print(f"Loading model from: {MODEL_PATH}")
# 모델 구조 정의 (학습 시와 동일하게!)
model = smp.Unet(
    encoder_name="resnet34",        # 학습 시 사용한 백본
    encoder_weights=None,           # state_dict를 로드할 것이므로 None
    in_channels=3,
    classes=NUM_CLASSES,
)

# 저장된 state_dict 로드
try:
    # GPU/CPU 호환성을 위해 map_location 사용
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the 'train.py' script was run and the model file exists.")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    print("Ensure the model architecture definition (Unet, encoder_name, classes) matches the saved weights.")
    exit()

model.to(DEVICE)
model.eval() # 모델을 평가 모드로 설정

# --- 5. 시각화 함수 ---
def tensor_to_numpy_image(tensor):
    """ PyTorch 텐서(C, H, W)를 시각화 가능한 NumPy 배열(H, W, C)로 변환 """
    img_np = tensor.cpu().numpy()
    if img_np.shape[0] == 3: # CHW -> HWC
      img_np = np.transpose(img_np, (1, 2, 0))

    # 정규화 되돌리기 (대략적)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255 # 0-1 범위를 0-255로
    return img_np.astype(np.uint8)

def visualize_and_save(image_tensor, pred_mask_np, gt_mask_np, filename, save_dir, alpha=0.6):
    """ 원본 이미지, 예측 마스크, 정답 마스크를 오버레이하여 시각화하고 저장 """
    os.makedirs(save_dir, exist_ok=True)

    # 원본 이미지 (텐서 -> 넘파이 변환 및 정규화 복원)
    image_np = tensor_to_numpy_image(image_tensor)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Matplotlib은 RGB

    # 예측 마스크 색상 적용
    pred_mask_color = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        pred_mask_color[pred_mask_np == class_id] = color

    # 정답 마스크 색상 적용
    gt_mask_color = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        gt_mask_color[gt_mask_np == class_id] = color

    # 오버레이 이미지 생성
    overlay_pred = cv2.addWeighted(image_rgb, 1, pred_mask_color, alpha, 0)
    overlay_gt = cv2.addWeighted(image_rgb, 1, gt_mask_color, alpha, 0)

    # 시각화 및 저장
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Prediction vs Ground Truth: {filename}", fontsize=16)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(overlay_pred)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis('off')

    axes[2].imshow(overlay_gt)
    axes[2].set_title("Ground Truth Overlay")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle 공간 확보
    save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_comparison.png")
    plt.savefig(save_path)
    plt.close(fig) # 메모리 해제

# --- 6. 테스트 데이터셋 평가 함수 ---
@torch.no_grad()
def test_model(model, test_loader, device, num_classes, result_dir, vis_dir, save_vis=True, max_vis=None):
    print("\n--- Evaluating model on test set ---")
    os.makedirs(result_dir, exist_ok=True)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    model.eval()

    # 메트릭 계산기 초기화 (torchmetrics 사용)
    # 배경 클래스(0)는 제외하고 계산하려면 ignore_index=0 추가
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    pixel_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='micro', ignore_index=0).to(device)
    # 참고: 'micro'는 전체 픽셀 기준 정확도, 'macro'는 클래스별 정확도 평균

    progress_bar = tqdm(test_loader, desc="Testing")
    vis_count = 0

    for batch in progress_bar:
        if batch is None: # collate_fn에서 빈 배치를 반환한 경우 건너뛰기
            continue

        # 데이터 로더 반환 값에 따라 언패킹 조정 필요
        # 여기서는 (image, mask, filename) 순서로 반환한다고 가정
        # Dataset에서 return_filename=True 로 설정해야 함
        images, masks, filenames = batch
        images = images.to(device)
        masks = masks.to(device) # (N, H, W) LongTensor

        # 예측
        outputs = model(images) # (N, C, H, W)

        # 예측 마스크 (가장 확률 높은 클래스)
        preds = torch.argmax(outputs, dim=1) # (N, H, W)

        # 메트릭 업데이트
        jaccard.update(preds, masks)
        pixel_acc.update(preds, masks)

        # 시각화 저장 (선택 사항)
        if save_vis and (max_vis is None or vis_count < max_vis):
            for i in range(images.size(0)):
                if max_vis is not None and vis_count >= max_vis:
                    break
                img_tensor = images[i].cpu()
                pred_mask_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_mask_np = masks[i].cpu().numpy().astype(np.uint8)
                filename = filenames[i]

                visualize_and_save(img_tensor, pred_mask_np, gt_mask_np, filename, vis_dir, alpha=VIS_ALPHA)
                vis_count += 1

                if max_vis is not None and vis_count >= max_vis:
                    # 시각화 저장 중단 메시지 표시 (한 번만)
                    if vis_count == max_vis:
                         print(f"\nReached maximum number of visualizations ({max_vis}). Stopping visualization saving.")
                    break # 내부 루프 탈출


    # 최종 메트릭 계산
    final_miou = jaccard.compute().item() # .item()으로 스칼라 값 추출
    final_accuracy = pixel_acc.compute().item()

    print("\n--- Test Results ---")
    print(f"Test mIoU (excluding background): {final_miou:.4f}")
    print(f"Test Pixel Accuracy (excluding background): {final_accuracy:.4f}")

    # 결과 파일 저장
    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("--- Test Metrics ---\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Test Dataset: {ANNOTATION_FILE_TEST}\n")
        f.write(f"Timestamp: {current_time}\n")
        f.write("-" * 20 + "\n")
        # MSE, MAE는 Segmentation에 일반적이지 않으므로 주석 처리. 필요시 계산 로직 추가.
        # f.write(f"Mean Squared Error (MSE): {mse_value:.4f}\n")
        # f.write(f"Mean Absolute Error (MAE): {mae_value:.4f}\n")
        f.write(f"Mean IoU (mIoU, excluding background): {final_miou:.4f}\n")
        f.write(f"Pixel Accuracy (excluding background): {final_accuracy:.4f}\n")
    print(f"Metrics saved to: {metrics_path}")
    if save_vis:
        print(f"Visualizations saved in: {VISUALIZATION_DIR}")

    # 계산기 상태 리셋 (필요시)
    jaccard.reset()
    pixel_acc.reset()

    return final_miou, final_accuracy

# --- 7. 메인 실행 ---
if __name__ == '__main__':
    print(f"Test Annotation File: {ANNOTATION_FILE_TEST}")
    print(f"Image Directory: {IMAGE_DIR_TEST}")
    print(f"Result Directory: {RESULT_DIR}")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # 테스트 데이터셋 및 로더 생성
    try:
        test_dataset = DocLayNetDataset(
            image_dir=IMAGE_DIR_TEST,
            annotation_file=ANNOTATION_FILE_TEST,
            class_map=CLASS_MAP,
            transforms=test_transforms,
            return_filename=True # 파일 이름 반환 활성화
        )
    except FileNotFoundError:
         print(f"Error: Test annotation file not found at {ANNOTATION_FILE_TEST}.")
         exit()
    except Exception as e:
         print(f"Error creating test dataset: {e}")
         exit()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 테스트는 섞지 않음
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_filter_none # None 값 처리 collate_fn 사용
    )

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Please check the annotation file and image directory.")
        exit()

    print(f"Test dataset size: {len(test_dataset)}") # 로더 생성 후 실제 사용될 데이터 수 확인

    # 테스트 실행
    start_time_test = time.time()
    test_model(model, test_loader, DEVICE, NUM_CLASSES, RESULT_DIR, VISUALIZATION_DIR,
               save_vis=SAVE_VISUALIZATIONS, max_vis=MAX_VISUALIZATIONS)
    end_time_test = time.time()
    total_test_time = end_time_test - start_time_test
    print(f"\nTotal Testing Time: {total_test_time:.2f} seconds")
    print("Testing finished.")

