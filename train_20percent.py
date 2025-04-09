# -*- coding: utf-8 -*-
import os
import json
import random # Subset 생성을 위해 추가
from PIL import Image
import numpy as np
import cv2
import time # 시간 측정용
import shutil # 폴더 관리용

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset # Subset 클래스 추가
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

# <<<--- 데이터셋 서브셋 비율 설정 --->>>
SUBSET_FRACTION = 0.2 # 사용할 데이터 비율 (예: 0.2 = 20%)
# <<<----------------------------------->>>

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
NUM_WORKERS = min(NUM_WORKERS, 4)

# --- 2. 데이터셋 클래스 정의  ---
class DocLayNetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, class_map, transforms=None):
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
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # 임시방편: 첫번째 이미지 정보로 대체
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
                if isinstance(ann['segmentation'], list):
                    if len(ann['segmentation']) > 0:
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:
                                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                                cv2.fillPoly(mask, [poly], color=class_label)
        if self.transforms:
            try:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                 print(f"Error during augmentation for image {img_info['file_name']}: {e}")
                 resize_only = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), ToTensorV2()])
                 augmented = resize_only(image=image, mask=mask)
                 image = augmented['image']
                 mask = augmented['mask']

        mask = mask.long()
        return image, mask

# --- 3. 데이터 변환 정의 (A.Affine 사용으로 수정됨) ---
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(1 - 0.05, 1 + 0.05),
        translate_percent=0.05,
        rotate=(-15, 15),
        p=0.5
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 4. 유틸리티 함수 ---
def validate_epoch(model, dataloader, criterion, metric_fn, device):
    model.eval()
    total_loss = 0.0
    metric_fn.reset()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            metric_fn.update(preds, masks)
            progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    final_metric = metric_fn.compute()
    return avg_loss, final_metric.item()

# ================================================================
# 메인 실행 블록 시작
# ================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("=" * 30)
    print("      Model Training Start      ")
    print("=" * 30)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Image Directory: {IMAGE_DIR}")
    print(f"Training Annotations: {ANNOTATION_FILE_TRAIN}")
    print(f"Validation Annotations: {ANNOTATION_FILE_VAL}")
    print(f"Model Save Directory: {MODEL_SAVE_DIR}")
    print(f"Subset Fraction: {SUBSET_FRACTION * 100:.1f}%") # 사용 비율 출력
    print("-" * 30)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print("-" * 30)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 5. 데이터셋 및 데이터 로더 생성 ---
    print("Creating full datasets...")
    try:
        full_train_dataset = DocLayNetDataset(
            image_dir=IMAGE_DIR,
            annotation_file=ANNOTATION_FILE_TRAIN,
            class_map=CLASS_MAP,
            transforms=train_transforms
        )
        full_val_dataset = DocLayNetDataset(
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

    print("Creating subsets...")
    # Train Subset 생성
    num_total_train = len(full_train_dataset)
    num_train_subset = int(num_total_train * SUBSET_FRACTION)
    train_indices = list(range(num_total_train))
    random.shuffle(train_indices) # 인덱스를 섞음
    train_subset_indices = train_indices[:num_train_subset]
    train_dataset = Subset(full_train_dataset, train_subset_indices)
    print(f"Using {len(train_dataset)} training samples out of {num_total_train} ({SUBSET_FRACTION*100:.1f}%)")

    # Validation Subset 생성
    num_total_val = len(full_val_dataset)
    num_val_subset = int(num_total_val * SUBSET_FRACTION)
    val_indices = list(range(num_total_val))
    random.shuffle(val_indices) # 인덱스를 섞음
    val_subset_indices = val_indices[:num_val_subset]
    val_dataset = Subset(full_val_dataset, val_subset_indices)
    print(f"Using {len(val_dataset)} validation samples out of {num_total_val} ({SUBSET_FRACTION*100:.1f}%)")

    # Subset을 이용하여 DataLoader 생성
    train_loader = DataLoader(
        train_dataset, # Subset 사용
        batch_size=BATCH_SIZE,
        shuffle=True, # Subset 내에서는 섞는 것이 좋음
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, # Subset 사용
        batch_size=BATCH_SIZE,
        shuffle=False, # 검증 시에는 섞지 않음
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # 데이터 로더 확인 (첫 배치 로드 시도)
    print("Testing dataloaders...")
    try:
        images, masks = next(iter(train_loader))
        print(f"Train loader OK. Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        images, masks = next(iter(val_loader))
        print(f"Validation loader OK. Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        del images, masks
    except Exception as e:
        print(f"Error testing dataloaders: {e}")
        exit()

    # --- 6. 모델 정의 (변경 없음) ---
    print("Defining the model...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.to(DEVICE)
    print("Model defined successfully.")

    # --- 7. 손실 함수, 옵티마이저, 메트릭 정의 (변경 없음) ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=0).to(DEVICE)

    # --- 8. 학습 루프 (변경 없음) ---
    print("\n--- Starting Training ---")
    best_val_metric = -1.0
    start_time_train = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # --- 학습 단계 ---
        model.train()
        epoch_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for images, masks in progress_bar_train:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
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
        torch.save(model.state_dict(), LAST_MODEL_PATH)
        print(f"  Last model saved to: {LAST_MODEL_PATH}")
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
