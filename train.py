# -*- coding: utf-8 -*-
import os
import json
import random
from PIL import Image
import numpy as np
import cv2
import time # 시간 측정용
import shutil # 폴더 관리용

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics # 평가 지표용

from tqdm import tqdm
import multiprocessing

# --- 1. 설정 (Configuration) ---
DATA_DIR = "DocLayNet_core"  # <<< 데이터셋 경로 수정!
ANNOTATION_FILE_TRAIN = os.path.join(DATA_DIR, "COCO", "train.json")
ANNOTATION_FILE_VAL = os.path.join(DATA_DIR, "COCO", "val.json") # 검증셋 어노테이션 추가
IMAGE_DIR = os.path.join(DATA_DIR, "PNG") # 학습/검증 이미지 폴더 동일

MODEL_SAVE_DIR = "saved_models" # 모델 저장 폴더
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "last_model.pth")

CLASS_NAMES = [
    'BACKGROUND', 'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4 # <<< 필요시 메모리에 맞게 조절
EPOCHS = 10 # <<< 학습 에포크 수 조절
LEARNING_RATE = 1e-4
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0
NUM_WORKERS = min(NUM_WORKERS, 4) # 예시: 최대 4개로 제한

# --- 2. 데이터셋 클래스 정의 ---
class DocLayNetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, class_map, transforms=None):
        self.image_dir = image_dir
        print(f"Loading annotations from: {annotation_file}")
        # COCO 로딩 메시지를 한 번만 출력하도록 수정
        try:
            self.coco = COCO(annotation_file)
        except Exception as e:
            print(f"Error loading COCO file {annotation_file}: {e}")
            raise # 오류 발생 시 프로그램 중지
        print(f"Annotations loaded successfully.")

        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.class_map = class_map
        self.transforms = transforms

        # 배경 클래스 제외하고 카테고리 ID 가져오기
        self.cat_ids = self.coco.getCatIds(catNms=[name for name in class_map.keys() if name != 'BACKGROUND'])
        self.cat_id_to_label = {cat_id: self.class_map[self.coco.loadCats(cat_id)[0]['name']]
                                for cat_id in self.cat_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        try:
            # 가끔 이미지 파일이 손상된 경우가 있을 수 있음
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # 빈 이미지와 마스크 반환 또는 다음 이미지 로드 시도 (여기서는 예외 발생 시 오류)
            # 좀 더 견고하게 만들려면 __getitem__에서 None을 반환하고 DataLoader의 collate_fn에서 처리 필요
            # 여기서는 간단히 오류 발생시키거나, 임의의 다른 이미지 반환 등 처리 가능
            # 임시방편: 첫번째 이미지 정보로 대체 (데이터셋에 따라 문제될 수 있음)
            img_id = self.image_ids[0]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            image = np.array(Image.open(img_path).convert("RGB"))


        original_h, original_w = image.shape[:2]

        mask = np.zeros((original_h, original_w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann['category_id']
            class_label = self.cat_id_to_label.get(category_id)

            if class_label is not None and 'segmentation' in ann:
                # DocLayNet은 RLE 또는 Polygon 형식을 가질 수 있음
                if isinstance(ann['segmentation'], list): # Polygon
                    if len(ann['segmentation']) > 0: # 비어있지 않은 segmentation만 처리
                        for seg in ann['segmentation']:
                            if len(seg) >= 6: # 유효한 폴리곤(최소 3점)인지 확인
                                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                                cv2.fillPoly(mask, [poly], color=class_label)
                            # else:
                            #     print(f"Warning: Invalid polygon with < 3 points in annotation {ann['id']} for image {img_info['file_name']}. Skipping segment.")
                # RLE 형식 처리 추가 (필요시)
                # elif isinstance(ann['segmentation'], dict): # RLE
                #     from pycocotools import mask as maskUtils
                #     h, w = ann['segmentation']['size']
                #     rle = maskUtils.frPyObjects([ann['segmentation']], h, w)
                #     m = maskUtils.decode(rle)
                #     # 여러 객체가 겹칠 때 순서대로 덮어쓰게 됨
                #     mask[m[:, :, 0] > 0] = class_label # m은 (h, w, 1) 형태일 수 있음

        # 변환 적용
        if self.transforms:
            try:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                 print(f"Error during augmentation for image {img_info['file_name']}: {e}")
                 # 변환 오류 시 대체 처리 (예: 원본 이미지 리사이즈만 적용)
                 resize_only = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), ToTensorV2()])
                 augmented = resize_only(image=image, mask=mask)
                 image = augmented['image']
                 mask = augmented['mask']


        mask = mask.long() # CrossEntropyLoss는 LongTensor 타입의 타겟 필요
        return image, mask

# --- 3. 데이터 변환 정의 (Albumentations) ---
# 학습용 변환 (Data Augmentation 포함 가능)
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), # 이전 코드
    A.Affine(
        scale=(1 - 0.05, 1 + 0.05),      # scale_limit=0.05 에 해당
        translate_percent=0.05,         # shift_limit=0.05 에 해당
        rotate=(-15, 15),               # rotate_limit=15 에 해당
        p=0.5                           # 동일한 적용 확률
        # Affine 변환의 다른 파라미터(shear 등)는 기본값 사용
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 검증/테스트용 변환 (Augmentation 없음)
val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# --- 4. 유틸리티 함수 ---
def validate_epoch(model, dataloader, criterion, metric_fn, device):
    """ 한 에포크 동안의 검증 손실과 메트릭(mIoU) 계산 """
    model.eval() # 모델을 평가 모드로 설정
    total_loss = 0.0
    metric_fn.reset() # 메트릭 계산기 초기화

    with torch.no_grad(): # 그래디언트 계산 비활성화
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device) # (N, H, W)

            outputs = model(images) # (N, C, H, W)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1) # (N, H, W)
            metric_fn.update(preds, masks)

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    final_metric = metric_fn.compute() # 최종 메트릭 계산
    return avg_loss, final_metric.item() # 메트릭 값만 반환

# ================================================================
# 메인 실행 블록 시작
# ================================================================
if __name__ == '__main__':
    # Windows 멀티프로세싱 지원
    multiprocessing.freeze_support()

    print("=" * 30)
    print("      Model Training Start      ")
    print("=" * 30)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Image Directory: {IMAGE_DIR}")
    print(f"Training Annotations: {ANNOTATION_FILE_TRAIN}")
    print(f"Validation Annotations: {ANNOTATION_FILE_VAL}")
    print(f"Model Save Directory: {MODEL_SAVE_DIR}")
    print("-" * 30)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print("-" * 30)

    # 모델 저장 폴더 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 5. 데이터셋 및 데이터 로더 생성 ---
    print("Creating datasets and dataloaders...")
    try:
        train_dataset = DocLayNetDataset(
            image_dir=IMAGE_DIR,
            annotation_file=ANNOTATION_FILE_TRAIN,
            class_map=CLASS_MAP,
            transforms=train_transforms
        )
        val_dataset = DocLayNetDataset(
            image_dir=IMAGE_DIR,
            annotation_file=ANNOTATION_FILE_VAL,
            class_map=CLASS_MAP,
            transforms=val_transforms
        )
    except FileNotFoundError as e:
        print(f"Error: Annotation file not found. Please check paths.")
        print(e)
        exit()
    except Exception as e:
        print(f"Error creating dataset: {e}")
        exit()


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), # GPU 사용 시 True로 설정하면 좋음
        # persistent_workers=True if NUM_WORKERS > 0 else False, # PyTorch 1.8 이상, 워커 유지 (선택 사항)
        # collate_fn=lambda b: tuple(zip(*filter(lambda x: x is not None, b))) # __getitem__에서 None 반환 시 필요
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, # 검증 시 배치 크기는 학습과 같거나 더 크게 설정 가능
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers=True if NUM_WORKERS > 0 else False,
        # collate_fn=lambda b: tuple(zip(*filter(lambda x: x is not None, b)))
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # 데이터 로더 확인 (첫 배치 로드 시도)
    print("Testing dataloaders...")
    try:
        images, masks = next(iter(train_loader))
        print(f"Train loader OK. Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        images, masks = next(iter(val_loader))
        print(f"Validation loader OK. Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        del images, masks # 메모리 확보
    except Exception as e:
        print(f"Error testing dataloaders: {e}")
        print(f"Consider setting NUM_WORKERS=0 if errors persist, especially on Windows.")
        exit()

    # --- 6. 모델 정의 ---
    print("Defining the model...")
    model = smp.Unet(
        encoder_name="resnet34",        # 예시 백본, 필요시 변경
        encoder_weights="imagenet",     # 사전 학습된 가중치 사용
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.to(DEVICE)
    print("Model defined successfully.")

    # --- 7. 손실 함수, 옵티마이저, 메트릭 정의 ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # mIoU 계산기 (배경 클래스 제외하고 계산)
    # task="multiclass" 필수, num_classes 지정 필수
    metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=0).to(DEVICE)

    # --- 8. 학습 루프 ---
    print("\n--- Starting Training ---")
    best_val_metric = -1.0 # 최적 mIoU 저장 변수 (mIoU는 높을수록 좋음)

    start_time_train = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # --- 학습 단계 ---
        model.train() # 모델을 학습 모드로 설정
        epoch_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for images, masks in progress_bar_train:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad() # 그래디언트 초기화
            outputs = model(images) # 예측
            loss = criterion(outputs, masks) # 손실 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트

            epoch_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- 검증 단계 ---
        val_loss, val_metric = validate_epoch(model, val_loader, criterion, metric, DEVICE)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU (excluding background): {val_metric:.4f}")
        print(f"  Epoch Duration: {epoch_duration:.2f} seconds")

        # --- 모델 저장 ---
        # 마지막 에포크 모델 저장
        torch.save(model.state_dict(), LAST_MODEL_PATH)
        print(f"  Last model saved to: {LAST_MODEL_PATH}")

        # 최적 모델 저장 (검증 mIoU 기준)
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ** Best model updated (mIoU: {best_val_metric:.4f}) and saved to: {BEST_MODEL_PATH} **")

    end_time_train = time.time()
    total_train_time = end_time_train - start_time_train
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_train_time:.2f} seconds")
    print(f"Best Validation mIoU achieved: {best_val_metric:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print(f"Last model saved at: {LAST_MODEL_PATH}")
