# -*- coding: utf-8 -*-
import os
import json
import random # Subset 생성을 위해 추가
from PIL import Image
import numpy as np
import cv2
import time
import datetime # 결과 폴더 이름 생성용
import shutil # 폴더 관리용

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset # Subset 클래스 추가
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
MODEL_PATH = os.path.join("saved_models", "best_model.pth") # <<< 로드할 최적 모델 파일 경로

# 테스트 데이터 정보
ANNOTATION_FILE_TEST = os.path.join(DATA_DIR, "COCO", "test.json") # <<< 테스트셋 어노테이션 경로
IMAGE_DIR_TEST = os.path.join(DATA_DIR, "PNG")

# <<<--- 데이터셋 서브셋 비율 설정 --->>>
SUBSET_FRACTION = 0.2 # 사용할 데이터 비율 (예: 0.2 = 20%)
# <<<----------------------------------->>>

# 결과 저장 폴더 설정
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"{current_time}_result_subset_{int(SUBSET_FRACTION*100)}pct" # 폴더 이름에 비율 명시
METRICS_FILE = os.path.join(RESULT_DIR, "metrics.txt")
VISUALIZATION_DIR = os.path.join(RESULT_DIR, "visualizations")

CLASS_NAMES = [
    'BACKGROUND', 'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0
NUM_WORKERS = min(NUM_WORKERS, 4)

SAVE_VISUALIZATIONS = True
MAX_VISUALIZATIONS = 20
VIS_ALPHA = 0.6

random.seed(42)
COLOR_MAP = {0: (0, 0, 0)}
for i in range(1, NUM_CLASSES):
    COLOR_MAP[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# --- 2. 데이터셋 클래스 (변경 없음, 단 return_filename=True 유지 확인) ---
class DocLayNetDataset(Dataset):
    # ... (train.py와 동일하게 유지, return_filename 파라미터 추가 가능성 고려) ...
    # 생성자에서 return_filename 인자를 받도록 수정하는 것이 좋음
    def __init__(self, image_dir, annotation_file, class_map, transforms=None, return_original=False, return_filename=False): # return_filename 추가
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
        self.return_filename = return_filename # 속성 설정

        self.cat_ids = self.coco.getCatIds(catNms=[name for name in class_map.keys() if name != 'BACKGROUND'])
        self.cat_id_to_label = {cat_id: self.class_map[self.coco.loadCats(cat_id)[0]['name']]
                                for cat_id in self.cat_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        filename = img_info['file_name'] # 파일 이름 가져오기

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
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
                if isinstance(ann['segmentation'], list):
                     if len(ann['segmentation']) > 0:
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:
                                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                                cv2.fillPoly(mask, [poly], color=class_label)

        transformed_image = None
        transformed_mask = None
        if self.transforms:
            try:
                augmented = self.transforms(image=image, mask=mask)
                transformed_image = augmented['image']
                transformed_mask = augmented['mask']
            except Exception as e:
                 print(f"Error during augmentation for image {img_info['file_name']}: {e}")
                 return None

        if transformed_mask is not None:
            transformed_mask = transformed_mask.long()

        # 반환 값 구성 (파일 이름 포함 확인)
        results = (transformed_image, transformed_mask)
        if self.return_original:
            results += (original_image,)
        if self.return_filename:
            results += (filename,) # 파일 이름 반환

        # 결과 튜플의 길이가 예상과 다르면 None 반환 (collate_fn에서 처리 위함)
        expected_len = 2 + (1 if self.return_original else 0) + (1 if self.return_filename else 0)
        if len(results) != expected_len or any(x is None for x in results[:2]): # 이미지 또는 마스크가 None이면 안됨
             print(f"Warning: Item creation failed for image {filename}. Returning None.")
             return None

        return results


# 데이터 로더에서 None 값을 걸러내기 위한 collate_fn
def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- 3. 데이터 변환 (변경 없음) ---
test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 4. 모델 로드 (변경 없음) ---
print("=" * 30)
print("      Model Testing Start       ")
print("=" * 30)
print(f"Loading model from: {MODEL_PATH}")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
)
try:
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()
model.to(DEVICE)
model.eval()

# --- 5. 시각화 함수 (변경 없음) ---
def tensor_to_numpy_image(tensor):
    img_np = tensor.cpu().numpy()
    if img_np.shape[0] == 3:
      img_np = np.transpose(img_np, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255
    return img_np.astype(np.uint8)

def visualize_and_save(image_tensor, pred_mask_np, gt_mask_np, filename, save_dir, alpha=0.6):
    os.makedirs(save_dir, exist_ok=True)
    image_np = tensor_to_numpy_image(image_tensor)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pred_mask_color = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        pred_mask_color[pred_mask_np == class_id] = color
    gt_mask_color = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        gt_mask_color[gt_mask_np == class_id] = color
    overlay_pred = cv2.addWeighted(image_rgb, 1, pred_mask_color, alpha, 0)
    overlay_gt = cv2.addWeighted(image_rgb, 1, gt_mask_color, alpha, 0)
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_comparison.png")
    plt.savefig(save_path)
    plt.close(fig)

# --- 6. 테스트 데이터셋 평가 함수 (변경 없음) ---
@torch.no_grad()
def test_model(model, test_loader, device, num_classes, result_dir, vis_dir, save_vis=True, max_vis=None):
    print("\n--- Evaluating model on test set ---")
    os.makedirs(result_dir, exist_ok=True)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    pixel_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='micro', ignore_index=0).to(device)
    progress_bar = tqdm(test_loader, desc="Testing")
    vis_count = 0

    for batch in progress_bar:
        if batch is None:
            continue

        # collate_fn이 None을 필터링하므로, 배치 내 아이템 개수 확인 불필요
        # 데이터 로더가 파일 이름을 반환하도록 Dataset 수정 필요
        images, masks, filenames = batch # 파일 이름을 받는다고 가정

        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        jaccard.update(preds, masks)
        pixel_acc.update(preds, masks)

        if save_vis and (max_vis is None or vis_count < max_vis):
            for i in range(images.size(0)):
                if max_vis is not None and vis_count >= max_vis:
                    break
                img_tensor = images[i].cpu()
                pred_mask_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_mask_np = masks[i].cpu().numpy().astype(np.uint8)
                filename = filenames[i] # 개별 파일 이름 사용
                visualize_and_save(img_tensor, pred_mask_np, gt_mask_np, filename, vis_dir, alpha=VIS_ALPHA)
                vis_count += 1
                if max_vis is not None and vis_count >= max_vis:
                     if vis_count == max_vis:
                         print(f"\nReached maximum number of visualizations ({max_vis}). Stopping visualization saving.")
                     break

    final_miou = jaccard.compute().item()
    final_accuracy = pixel_acc.compute().item()
    print("\n--- Test Results ---")
    print(f"Test mIoU (excluding background): {final_miou:.4f}")
    print(f"Test Pixel Accuracy (excluding background): {final_accuracy:.4f}")

    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("--- Test Metrics ---\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Test Dataset: {ANNOTATION_FILE_TEST}\n")
        f.write(f"Subset Fraction: {SUBSET_FRACTION * 100:.1f}%\n") # 사용 비율 기록
        f.write(f"Timestamp: {current_time}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean IoU (mIoU, excluding background): {final_miou:.4f}\n")
        f.write(f"Pixel Accuracy (excluding background): {final_accuracy:.4f}\n")
    print(f"Metrics saved to: {metrics_path}")
    if save_vis:
        print(f"Visualizations saved in: {VISUALIZATION_DIR}")

    jaccard.reset()
    pixel_acc.reset()
    return final_miou, final_accuracy

# --- 7. 메인 실행 ---
if __name__ == '__main__':
    print(f"Test Annotation File: {ANNOTATION_FILE_TEST}")
    print(f"Image Directory: {IMAGE_DIR_TEST}")
    print(f"Result Directory: {RESULT_DIR}")
    print(f"Subset Fraction: {SUBSET_FRACTION * 100:.1f}%") # 사용 비율 출력
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # 테스트 데이터셋 및 로더 생성
    try:
        # 전체 테스트 데이터셋 로드
        full_test_dataset = DocLayNetDataset(
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

    # 테스트 Subset 생성
    num_total_test = len(full_test_dataset)
    num_test_subset = int(num_total_test * SUBSET_FRACTION)
    test_indices = list(range(num_total_test))
    random.shuffle(test_indices) # 인덱스 섞기
    test_subset_indices = test_indices[:num_test_subset]
    test_dataset = Subset(full_test_dataset, test_subset_indices) # Subset 생성
    print(f"Using {len(test_dataset)} testing samples out of {num_total_test} ({SUBSET_FRACTION*100:.1f}%)")


    if len(test_dataset) == 0: # Subset 크기 확인
        print("Error: Test dataset subset is empty. Please check the annotation file and subset fraction.")
        exit()

    # Subset으로 DataLoader 생성
    test_loader = DataLoader(
        test_dataset, # Subset 사용
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_filter_none # None 처리 collate_fn 사용
    )

    # 테스트 실행
    start_time_test = time.time()
    test_model(model, test_loader, DEVICE, NUM_CLASSES, RESULT_DIR, VISUALIZATION_DIR,
               save_vis=SAVE_VISUALIZATIONS, max_vis=MAX_VISUALIZATIONS)
    end_time_test = time.time()
    total_test_time = end_time_test - start_time_test
    print(f"\nTotal Testing Time: {total_test_time:.2f} seconds")
    print("Testing finished.")
