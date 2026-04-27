# -*- coding: utf-8 -*-

import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    !nvidia-smi
else:
    print("No GPU detected. In Colab: Runtime → Change runtime type → GPU")

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

# TODO: change this if your Drive folder is different
DRIVE_DATA_ROOT = Path('/content/drive/MyDrive/AAE5303/data')
# Labels are usually inside DRIVE_DATA_ROOT / 'AMtown_label'
DRIVE_LABEL_ROOT = DRIVE_DATA_ROOT / 'AMtown_label'

# You said you will store cmap.py in Drive
DRIVE_CMAP_PATH = DRIVE_DATA_ROOT / 'cmap.py'

print('DRIVE_DATA_ROOT :', DRIVE_DATA_ROOT)
print('DRIVE_LABEL_ROOT:', DRIVE_LABEL_ROOT)
print('DRIVE_CMAP_PATH :', DRIVE_CMAP_PATH)

assert DRIVE_DATA_ROOT.exists(), f"Not found: {DRIVE_DATA_ROOT}"
assert DRIVE_CMAP_PATH.exists(), f"Not found: {DRIVE_CMAP_PATH} (upload cmap.py to Drive)"
assert DRIVE_LABEL_ROOT.exists(), (
    f"Not found: {DRIVE_LABEL_ROOT}. You must upload AMtown labels to Drive under AMtown_label/..."
)

def split_dirs(split_id: str):
    img_dir = DRIVE_DATA_ROOT / f'interval5_AMtown{split_id}' / 'interval5_CAM'
    label_id_dir = DRIVE_LABEL_ROOT / f'interval5_AMtown{split_id}' / 'interval5_CAM_label_id'
    label_color_dir = DRIVE_LABEL_ROOT / f'interval5_AMtown{split_id}' / 'interval5_CAM_label_color'
    return img_dir, label_id_dir, label_color_dir

for split_id in ['01', '02', '03']:
    img_dir, label_id_dir, label_color_dir = split_dirs(split_id)

    print(f'\n=== AMtown{split_id} ===')
    print('Images:', img_dir, 'exists=', img_dir.exists())
    if img_dir.exists():
        print('  image count (jpg):', len(list(img_dir.glob('*.jpg'))))

    print('Label ID:', label_id_dir, 'exists=', label_id_dir.exists())
    if label_id_dir.exists():
        print('  id mask count (png):', len(list(label_id_dir.glob('*.png'))))

    print('Label Color:', label_color_dir, 'exists=', label_color_dir.exists())
    if label_color_dir.exists():
        print('  color mask count (png):', len(list(label_color_dir.glob('*.png'))))

print('\nIf labels are missing, upload AMtown_label/interval5_AMtownXX/... into Drive and re-run.')

# Commented out IPython magic to ensure Python compatibility.
import os

REPO_URL = 'https://github.com/milesial/Pytorch-UNet.git'
PROJECT_DIR = '/content/Pytorch-UNet'

if not os.path.exists(PROJECT_DIR):
    !git clone --depth 1 $REPO_URL $PROJECT_DIR

# %cd $PROJECT_DIR
!ls -la

import importlib
import subprocess
import sys

def ensure_import(module_name: str, pip_name=None):
    try:
        importlib.import_module(module_name)
    except ImportError:
        name = pip_name or module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])

ensure_import("tqdm")
ensure_import("PIL", pip_name="Pillow")
ensure_import("matplotlib")

print("Dependencies OK")

import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

print('Imports OK')

import importlib.util

IGNORE_INDEX = 255
IGNORE_RGB = np.array([255, 0, 255], dtype=np.uint8)  # magenta

def load_cmap(cmap_path):
    spec = importlib.util.spec_from_file_location('amtown_cmap', str(cmap_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'cmap'):
        raise ValueError(f'No `cmap` dict in {cmap_path}')
    return {int(k): v for k, v in mod.cmap.items()}

CMAP_RAW = load_cmap(DRIVE_CMAP_PATH)
ALL_CLASS_IDS = sorted(CMAP_RAW.keys())
NUM_TOTAL_CLASSES = len(ALL_CLASS_IDS)

def _is_empty_name(meta) -> bool:
    name = meta.get('name')
    return name is None or str(name).strip() == ''

EMPTY_NAME_CLASS_IDS = sorted([cid for cid in ALL_CLASS_IDS if _is_empty_name(CMAP_RAW[cid])])
NAMED_CLASS_IDS = sorted([cid for cid in ALL_CLASS_IDS if cid not in set(EMPTY_NAME_CLASS_IDS)])

print('Total classes in cmap.py:', NUM_TOTAL_CLASSES)
print('Removed (empty-name) class IDs:', EMPTY_NAME_CLASS_IDS)
print('Remaining named class IDs:', NAMED_CLASS_IDS)

print('\n{:<4} {:<24} {:<18} {:<8}'.format('ID', 'Name', 'RGB', 'Status'))
print('-' * 70)
for cid in ALL_CLASS_IDS:
    meta = CMAP_RAW[cid]
    raw_name = (meta.get('name') or '').strip()
    name = raw_name if raw_name else f'class_{cid}'
    rgb = meta.get('RGB')
    status = 'KEEP' if cid in NAMED_CLASS_IDS else 'DROP'
    print('{:<4} {:<24} {:<18} {:<8}'.format(cid, name, str(rgb), status))

# ID -> RGB table for visualization (size 256 so it can also index IGNORE_INDEX=255)
ID2RGB_256 = np.zeros((256, 3), dtype=np.uint8)
for cid, meta in CMAP_RAW.items():
    rgb = np.array(meta['RGB'], dtype=np.uint8)
    ID2RGB_256[int(cid)] = rgb
for cid in EMPTY_NAME_CLASS_IDS:
    ID2RGB_256[int(cid)] = IGNORE_RGB
ID2RGB_256[IGNORE_INDEX] = IGNORE_RGB

# RGB -> ID lookup for color labels (unknown colors map to IGNORE_INDEX)
RGB_LUT_ALL = np.full((256, 256, 256), IGNORE_INDEX, dtype=np.uint8)
for cid, meta in CMAP_RAW.items():
    r, g, b = meta['RGB']
    RGB_LUT_ALL[int(r), int(g), int(b)] = int(cid)

KNOWN_ID_SET = set(ALL_CLASS_IDS)

def clean_id_mask(mask_id: np.ndarray) -> np.ndarray:
    """Validate and clean an ID mask.

    - Ensures IDs are either in cmap.py OR IGNORE_INDEX.
    - Maps empty-name class IDs to IGNORE_INDEX.
    """
    if mask_id.ndim != 2:
        raise ValueError(f'Expected a 2D mask, got shape {mask_id.shape}')

    unique = np.unique(mask_id)
    bad = [int(v) for v in unique.tolist() if int(v) not in KNOWN_ID_SET and int(v) != IGNORE_INDEX]
    if bad:
        raise ValueError(f'Found unknown class IDs in mask: {bad[:10]} (showing up to 10)')

    cleaned = mask_id.copy()
    for cid in EMPTY_NAME_CLASS_IDS:
        cleaned[cleaned == cid] = IGNORE_INDEX
    return cleaned

def colorize_id_mask(mask_id: np.ndarray) -> np.ndarray:
    """Convert (H,W) ID mask into (H,W,3) RGB for visualization."""
    return ID2RGB_256[mask_id]

print('\nHelpers ready')

import shutil

@dataclass
class SplitSource:
    split_id: str
    images_dir: Path
    label_id_dir: Path
    label_color_dir: Path

def resolve_split_source(split_id: str) -> SplitSource:
    img_dir, label_id_dir, label_color_dir = split_dirs(split_id)
    return SplitSource(
        split_id=split_id,
        images_dir=img_dir,
        label_id_dir=label_id_dir,
        label_color_dir=label_color_dir,
    )

def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == 'symlink':
        os.symlink(src, dst)
        return
    if mode == 'copy':
        shutil.copy2(src, dst)
        return
    raise ValueError(f'Unknown mode: {mode}')

def convert_color_mask_to_id(mask_path: Path, rgb_lut: np.ndarray) -> np.ndarray:
    rgb = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8)
    ids = rgb_lut[rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    unknown = ids == 255
    if unknown.any():
        bad_rgb = rgb[unknown][0].tolist()
        raise ValueError(f'Unknown label color {bad_rgb} in {mask_path}')
    return ids

def prepare_flat_dataset(
    split_ids: List[str],
    out_images_dir: Path,
    out_masks_dir: Path,
    mode: str = 'symlink',
    label_type: str = 'auto',  # 'auto' | 'id' | 'color'
    prefix_split: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """Prepare a flat dataset (images + ID masks) for one or more splits."""

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    paired = 0
    missing_masks = 0

    for split_id in split_ids:
        src = resolve_split_source(split_id)

        if not src.images_dir.exists():
            raise FileNotFoundError(f'Images dir missing: {src.images_dir}')

        use_id = False
        use_color = False
        if label_type == 'id':
            use_id = True
        elif label_type == 'color':
            use_color = True
        elif label_type == 'auto':
            use_id = src.label_id_dir.exists()
            use_color = not use_id
        else:
            raise ValueError(f'Unknown label_type: {label_type}')

        if use_id and not src.label_id_dir.exists():
            raise FileNotFoundError(f'Label ID dir missing: {src.label_id_dir}')
        if use_color and not src.label_color_dir.exists():
            raise FileNotFoundError(f'Label COLOR dir missing: {src.label_color_dir}')

        image_files = sorted(src.images_dir.glob('*.jpg'))
        if not image_files:
            image_files = sorted(src.images_dir.glob('*.png'))
        if not image_files:
            raise RuntimeError(f'No images found in: {src.images_dir}')

        if verbose:
            print('\n' + '=' * 80)
            print(f'Split AMtown{split_id}: images={len(image_files)} | label_type=' + ('id' if use_id else 'color'))
            print('images_dir:', src.images_dir)
            print('labels_dir:', src.label_id_dir if use_id else src.label_color_dir)
            print('=' * 80)

        for img_path in tqdm(image_files, desc=f'Preparing AMtown{split_id}', unit='img'):
            stem = img_path.stem
            out_stem = f'AMtown{split_id}_{stem}' if prefix_split else stem

            # Copy/symlink image
            out_img = out_images_dir / f'{out_stem}{img_path.suffix}'
            if not out_img.exists():
                link_or_copy(img_path, out_img, mode)

            # Find and prepare mask
            if use_id:
                candidates = list(src.label_id_dir.glob(stem + '.*'))
                if len(candidates) != 1:
                    missing_masks += 1
                    continue
                mask_path = candidates[0]
                out_mask = out_masks_dir / f'{out_stem}{mask_path.suffix}'
                if not out_mask.exists():
                    link_or_copy(mask_path, out_mask, mode)
            else:
                candidates = list(src.label_color_dir.glob(stem + '.*'))
                if len(candidates) != 1:
                    missing_masks += 1
                    continue
                mask_path = candidates[0]
                out_mask = out_masks_dir / f'{out_stem}.png'
                if not out_mask.exists():
                    ids = convert_color_mask_to_id(mask_path, RGB_LUT)
                    Image.fromarray(ids, mode='L').save(out_mask)

            paired += 1
            total_images += 1

    # Final consistency check
    img_stems = {p.stem for p in out_images_dir.glob('*') if p.is_file() and not p.name.startswith('.')}
    mask_stems = {p.stem for p in out_masks_dir.glob('*') if p.is_file() and not p.name.startswith('.')}

    stats = {
        'images_out': len(img_stems),
        'masks_out': len(mask_stems),
        'paired': len(img_stems & mask_stems),
        'missing_masks': len(img_stems - mask_stems),
        'missing_images': len(mask_stems - img_stems),
        'missing_masks_during_prepare': int(missing_masks),
    }

    if verbose:
        print('\nPrepared dataset summary')
        for k, v in stats.items():
            print(f'  {k}: {v}')

    return stats

print('Preparation functions ready')

# Prepared output location (Colab runtime is fastest)
PREP_ROOT = Path('/content/amtown_prepared')

TRAIN_IMAGES_DIR = PREP_ROOT / 'train_images'
TRAIN_MASKS_DIR = PREP_ROOT / 'train_masks'
TEST_IMAGES_DIR = PREP_ROOT / 'test_images'
TEST_MASKS_DIR = PREP_ROOT / 'test_masks'

# Choose preparation mode:
# - 'symlink': fast, no duplication (recommended)
# - 'copy': duplicates files into PREP_ROOT (may be faster than Drive reads if you have enough disk)
PREP_MODE = 'symlink'

TRAIN_SPLITS = ['01', '02']
TEST_SPLITS = ['03']

print('PREP_ROOT:', PREP_ROOT)
print('PREP_MODE:', PREP_MODE)
print('TRAIN_SPLITS:', TRAIN_SPLITS)
print('TEST_SPLITS :', TEST_SPLITS)

# Clean prepared folders if you want to re-run from scratch
# (Safe: only deletes inside PREP_ROOT)
if PREP_ROOT.exists():
    shutil.rmtree(PREP_ROOT)

train_stats = prepare_flat_dataset(
    split_ids=TRAIN_SPLITS,
    out_images_dir=TRAIN_IMAGES_DIR,
    out_masks_dir=TRAIN_MASKS_DIR,
    mode=PREP_MODE,
    label_type='auto',
    prefix_split=True,
)

test_stats = prepare_flat_dataset(
    split_ids=TEST_SPLITS,
    out_images_dir=TEST_IMAGES_DIR,
    out_masks_dir=TEST_MASKS_DIR,
    mode=PREP_MODE,
    label_type='auto',
    prefix_split=True,
)

train_images = sorted(TRAIN_IMAGES_DIR.glob('*.jpg'))
if not train_images:
    train_images = sorted(TRAIN_IMAGES_DIR.glob('*.png'))
assert train_images, f'No prepared images found in {TRAIN_IMAGES_DIR}'

img_path = random.choice(train_images)
mask_path = TRAIN_MASKS_DIR / f'{img_path.stem}.png'
assert mask_path.exists(), f'Mask missing for {img_path.stem}: {mask_path}'

img = Image.open(img_path).convert('RGB')
mask_id = np.array(Image.open(mask_path), dtype=np.uint8)
mask_rgb = colorize_id_mask(mask_id)

present = sorted({int(x) for x in np.unique(mask_id)})
present_names = [(cid, CMAP_RAW[cid].get('name') or f'class_{cid}') for cid in present]

print('Example image:', img_path)
print('Example mask :', mask_path)
print('Unique class IDs in mask:', present)
print('Class names:', present_names)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img)
ax[0].set_title('Image')
ax[0].axis('off')

ax[1].imshow(mask_id, cmap='tab20', vmin=0, vmax=NUM_TOTAL_CLASSES - 1)
ax[1].set_title('Mask (IDs)')
ax[1].axis('off')

ax[2].imshow(mask_rgb)
ax[2].set_title('Mask (Colorized via cmap.py)')
ax[2].axis('off')
plt.show()

from torch.utils.data import Dataset

class AMtownFlatDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1.0
        self.scale = float(scale)

        # Build IDs from image files
        img_files = [p for p in self.images_dir.iterdir() if p.is_file() and not p.name.startswith('.')]
        img_files = [p for p in img_files if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
        self.ids = sorted([p.stem for p in img_files])
        if not self.ids:
            raise RuntimeError(f'No images found in {self.images_dir}')

        # Fast value→index mapping (identity for 0..25, but we still validate)
        self.valid_ids = np.arange(NUM_TOTAL_CLASSES, dtype=np.uint8)  # Fixed to NUM_TOTAL_CLASSES
        self.value_to_index = np.full((256,), 255, dtype=np.uint8)
        for v in self.valid_ids:
            self.value_to_index[int(v)] = int(v)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, stem: str) -> Image.Image:
        # accept multiple extensions
        candidates = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            p = self.images_dir / (stem + ext)
            if p.exists():
                candidates.append(p)
        if len(candidates) != 1:
            raise FileNotFoundError(f'Expected 1 image for {stem}, got {candidates}')
        return Image.open(candidates[0]).convert('RGB')

    def _load_mask(self, stem: str) -> Image.Image:
        p = self.masks_dir / (stem + '.png')
        if not p.exists():
            raise FileNotFoundError(f'Mask not found: {p}')
        # Keep as grayscale (IDs)
        return Image.open(p)

    def _resize(self, img: Image.Image, is_mask: bool) -> Image.Image:
        if self.scale == 1.0:
            return img
        w, h = img.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        if nw <= 0 or nh <= 0:
            raise ValueError('Scale too small, resized size is zero')
        resample = Image.NEAREST if is_mask else Image.BICUBIC
        return img.resize((nw, nh), resample=resample)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]

        img = self._load_image(stem)
        mask = self._load_mask(stem)

        if img.size != mask.size:
            raise ValueError(f'Size mismatch for {stem}: image={img.size}, mask={mask.size}')

        img = self._resize(img, is_mask=False)
        mask = self._resize(mask, is_mask=True)

        img_np = np.array(img, dtype=np.float32)  # HWC, 0..255
        img_np = img_np.transpose(2, 0, 1) / 255.0  # CHW, 0..1

        mask_np = np.array(mask, dtype=np.uint8)
        if mask_np.ndim != 2:
            raise ValueError(f'Expected 2D mask, got shape {mask_np.shape} for {stem}')

        mapped = self.value_to_index[mask_np]
        if (mapped == 255).any():
            bad = int(mask_np[mapped == 255][0])
            raise ValueError(f'Unknown class id {bad} in mask for {stem}')

        return {
            'image': torch.from_numpy(img_np).float().contiguous(),
            'mask': torch.from_numpy(mapped.astype(np.int64)).long().contiguous(),
            'stem': stem,
        }

# Training configuration (you can tune these)
SCALE = 0.25
BATCH_SIZE = 8
VAL_PERCENT = 0.1
NUM_WORKERS = 0
SEED = 0

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

full_train_ds = AMtownFlatDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, scale=SCALE)
n_val = int(len(full_train_ds) * VAL_PERCENT)
n_train = len(full_train_ds) - n_val
train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

test_ds = AMtownFlatDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, scale=SCALE)

loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)
test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_args)

print('Train samples:', len(train_ds))
print('Val samples  :', len(val_ds))
print('Test samples :', len(test_ds))

def update_confusion_matrix(conf: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int):
    mask = (target >= 0) & (target < num_classes)
    label = num_classes * target[mask].astype('int64') + pred[mask].astype('int64')
    count = np.bincount(label, minlength=num_classes ** 2)
    conf += count.reshape(num_classes, num_classes)

def metrics_from_confusion(conf: np.ndarray, ignore_background: bool = False) -> Dict:
    num_classes = conf.shape[0]
    diag = np.diag(conf).astype(np.float64)
    sum_row = conf.sum(axis=1).astype(np.float64)
    sum_col = conf.sum(axis=0).astype(np.float64)
    total = conf.sum().astype(np.float64)

    union = sum_row + sum_col - diag
    iou = diag / (union + 1e-10)
    dice = 2 * diag / (sum_row + sum_col + 1e-10)
    acc = diag / (sum_row + 1e-10)

    valid = sum_row > 0
    classes = np.arange(num_classes)

    if ignore_background:
        valid = valid & (classes != 0)

    miou = float(np.nanmean(iou[valid])) if valid.any() else float('nan')
    mdice = float(np.nanmean(dice[valid])) if valid.any() else float('nan')
    macc = float(np.nanmean(acc[valid])) if valid.any() else float('nan')
    pix_acc = float(diag.sum() / (total + 1e-10))

    return {
        'pixel_accuracy': pix_acc,
        'mean_iou': miou,
        'mean_dice': mdice,
        'mean_accuracy': macc,
        'per_class_iou': [float(x) for x in iou.tolist()],
        'per_class_dice': [float(x) for x in dice.tolist()],
        'per_class_accuracy': [float(x) for x in acc.tolist()],
        'valid_classes': [int(c) for c in classes[valid].tolist()],
    }

print('Metrics helpers ready')

def compute_multiclass_dice(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    probs = F.softmax(pred_logits, dim=1)
    pred_onehot = F.one_hot(probs.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    return float(1.0 - dice_loss(pred_onehot[:, 1:], target_onehot[:, 1:], multiclass=True))

@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, use_amp: bool) -> Dict[str, float]:
    model.eval()
    dice_scores = []
    for batch in tqdm(loader, desc='Validating', leave=False):
        images = batch['image'].to(device=device, dtype=torch.float32)
        masks = batch['mask'].to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
        dice_scores.append(compute_multiclass_dice(logits, masks, num_classes))
    model.train()
    return {'val_dice_no_bg': float(np.mean(dice_scores)) if dice_scores else float('nan')}

def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int = 30, # 默认改为 30
    lr: float = 1e-6,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    checkpoint_dir: Path = Path('/content/drive/MyDrive/unet_amtown_checkpoints'),
    class_weights: torch.Tensor = None, # 接收权重
) -> Dict[str, List[float]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 引入权重，并忽略 255 标签
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    history = {'train_loss': [], 'val_dice_no_bg': [], 'lr': []}
    import random

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit='batch')

        for batch in pbar:
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.long)

            # --- 在线数据增强 ---
            if random.random() > 0.5:
                images = torch.flip(images, dims=[3]) # 水平
                masks = torch.flip(masks, dims=[2])
            if random.random() > 0.5:
                images = torch.flip(images, dims=[2]) # 垂直
                masks = torch.flip(masks, dims=[1])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                ce = criterion(logits, masks)
                probs = F.softmax(logits, dim=1)
                masks_1h = F.one_hot(masks, num_classes).permute(0, 3, 1, 2).float()
                dloss = dice_loss(probs, masks_1h, multiclass=True)
                loss = ce + dloss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))
            pbar.set_postfix({'loss': float(loss.item())})

        val_metrics = validate(model, val_loader, device, num_classes, use_amp)
        val_dice = val_metrics['val_dice_no_bg']
        scheduler.step(val_dice)

        # 保存权重
        ckpt_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
        state = model.state_dict()
        state['mask_values'] = list(range(num_classes))
        torch.save(state, ckpt_path)

        history['train_loss'].append(float(np.mean(epoch_losses)))
        history['val_dice_no_bg'].append(val_dice)
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        print(f"Epoch {epoch}: loss={history['train_loss'][-1]:.4f} | dice={val_dice:.4f}")

    return history

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

EPOCHS = 30
LR = 1e-6
CHECKPOINT_DIR = Path('/content/drive/MyDrive/unet_amtown_checkpoints')

# 计算类别权重：给稀有小类别（路、车、设备）更多关注
weights = torch.ones(NUM_TOTAL_CLASSES, dtype=torch.float32, device=DEVICE)
weights[0] = 0.1
weights[13] = 0.5
weights[2] = 5.0
weights[3] = 5.0
weights[15] = 10.0
weights[20] = 10.0

model = UNet(n_channels=3, n_classes=NUM_TOTAL_CLASSES, bilinear=False).to(DEVICE)

# 开始最终冲刺训练
history = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    num_classes=NUM_TOTAL_CLASSES,
    epochs=EPOCHS,
    lr=LR,
    use_amp=USE_AMP,
    checkpoint_dir=CHECKPOINT_DIR,
    class_weights=weights,
)

import re

ckpts = list(CHECKPOINT_DIR.glob('checkpoint_epoch*.pth'))
assert ckpts, f'No checkpoints found in {CHECKPOINT_DIR}'

def epoch_num(p: Path) -> int:
    m = re.search(r'checkpoint_epoch(\d+)\.pth', p.name)
    return int(m.group(1)) if m else -1

ckpts = sorted(ckpts, key=epoch_num)
MODEL_PATH = ckpts[-1]
print('Using checkpoint:', MODEL_PATH)

# Load model
state = torch.load(MODEL_PATH, map_location=DEVICE)
# Changed NUM_CLASSES to NUM_TOTAL_CLASSES below
mask_values = state.pop('mask_values', list(range(NUM_TOTAL_CLASSES)))
assert mask_values == list(range(NUM_TOTAL_CLASSES)), 'Unexpected mask_values mapping'

model = UNet(n_channels=3, n_classes=NUM_TOTAL_CLASSES, bilinear=False).to(DEVICE)
model.load_state_dict(state)
model.eval();
print('Model loaded')

@torch.inference_mode()
def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, use_amp: bool, max_images=None) -> Dict:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    seen = 0
    for batch in tqdm(loader, desc='Testing', unit='img'):
        images = batch['image'].to(device=device, dtype=torch.float32)
        targets = batch['mask'].numpy()  # (B,H,W)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()  # (B,H,W)

        for p, t in zip(preds, targets):
            update_confusion_matrix(conf, p, t, num_classes)
            seen += 1
            if max_images is not None and seen >= max_images:
                break
        if max_images is not None and seen >= max_images:
            break

    metrics_all = metrics_from_confusion(conf, ignore_background=False)
    metrics_no_bg = metrics_from_confusion(conf, ignore_background=True)

    return {
        'num_classes': num_classes,
        'images_evaluated': int(seen),
        'metrics_all': metrics_all,
        'metrics_no_bg': metrics_no_bg,
        'confusion_matrix': conf,
    }

MAX_TEST_IMAGES = None  # set e.g. 100 for a fast check
# Changed NUM_CLASSES to NUM_TOTAL_CLASSES below
test_report = evaluate_on_loader(model, test_loader, DEVICE, NUM_TOTAL_CLASSES, USE_AMP, max_images=MAX_TEST_IMAGES)

print('\nTest metrics (ALL classes)')
print('  pixel_accuracy:', test_report['metrics_all']['pixel_accuracy'])
print('  mean_iou      :', test_report['metrics_all']['mean_iou'])
print('  mean_dice     :', test_report['metrics_all']['mean_dice'])

print('\nTest metrics (IGNORE background class 0)')
print('  pixel_accuracy:', test_report['metrics_no_bg']['pixel_accuracy'])
print('  mean_iou      :', test_report['metrics_no_bg']['mean_iou'])
print('  mean_dice     :', test_report['metrics_no_bg']['mean_dice'])

valid = set(test_report['metrics_all']['valid_classes'])
iou = test_report['metrics_all']['per_class_iou']
dice = test_report['metrics_all']['per_class_dice']

print('{:<4} {:<24} {:>10} {:>10}'.format('ID', 'Name', 'IoU', 'Dice'))
print('-' * 52)
for cid in range(NUM_TOTAL_CLASSES):
    if cid not in valid:
        continue
    name = CMAP_RAW.get(cid, {}).get('name') or f'class_{cid}'
    print('{:<4} {:<24} {:>10.4f} {:>10.4f}'.format(cid, name, float(iou[cid]), float(dice[cid])))

REPORT_PATH = CHECKPOINT_DIR / 'amtown03_evaluation_report.json'

serializable = dict(test_report)
serializable['confusion_matrix'] = serializable['confusion_matrix'].tolist()
serializable['model_path'] = str(MODEL_PATH)
serializable['scale'] = SCALE
serializable['train_splits'] = TRAIN_SPLITS
serializable['test_splits'] = TEST_SPLITS

with open(REPORT_PATH, 'w') as f:
    json.dump(serializable, f, indent=2)

print('Saved report to:', REPORT_PATH)

@torch.inference_mode()
def visualize_predictions(model: nn.Module, dataset: Dataset, device: torch.device, n: int = 3):
    indices = random.sample(range(len(dataset)), k=min(n, len(dataset)))
    for idx in indices:
        sample = dataset[idx]
        img = sample['image'].unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = sample['mask'].numpy().astype(np.uint8)

        logits = model(img)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        img_vis = (sample['image'].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gt_rgb = colorize_id_mask(gt)
        pred_rgb = colorize_id_mask(pred)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(img_vis)
        ax[0].set_title(f'Input ({sample["stem"]})')
        ax[0].axis('off')

        ax[1].imshow(gt_rgb)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')

        ax[2].imshow(pred_rgb)
        ax[2].set_title('Prediction')
        ax[2].axis('off')
        plt.show()

visualize_predictions(model, test_ds, DEVICE, n=3)