"""Create the three self-contained, presentation-ready ROI baseline notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
EXPECTED_MANIFEST_SHA = "8ed4c05240a2d23e6c254ee74adf13af48ff45a4418f1e2024bd38498b02cab1"
EXPECTED_CLASSES = ["no_visible_inflammation", "mild_inflammation", "marked_inflammation"]


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def ascii_dashes(text: str) -> str:
    return text.translate(str.maketrans({
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u00ad": "-",
    }))


def shared_cells(model_id: str, model_title: str, initialization: str):
    cells = [
        md(f"""
# {model_title}

## Research question

This notebook evaluates **{model_title}** on the locked train/validation gingival ROI dataset. The task is three-class appearance classification: no visible inflammation, mild inflammation, and marked inflammation. It is a localized image-classification experiment, not a patient diagnosis.

The experiment uses the same ROI manifest, image processing, ImageNet normalization, training recipe, validation metrics, and checkpoint rules as notebooks 6.1–6.3. Notebook 6.1 uses a small CNN initialized randomly; 6.2 uses a manually implemented ResNet18 initialized randomly; 6.3 uses the same ResNet18 architecture initialized with ImageNet weights. Only the model architecture or initialization stated for this notebook changes.

**Expected result:** a reproducible validation report, learning curves, sample predictions, and resumable model checkpoints. No accuracy improvement is assumed in advance.
"""),
        md("""
## 1. Set the experiment configuration

These settings define one screening run. The train/validation split is fixed upstream. Change `RUN_TRAINING` to `True` after forking this notebook and enabling a GPU in Kaggle. `RESUME_CHECKPOINT` can point to a checkpoint attached from an earlier Kaggle run.
"""),
        code(f"""
from pathlib import Path
import os, json, hashlib, random, time, math, shutil

EXPERIMENT_ID = "{model_id}"
MODEL_TITLE = "{model_title}"
INITIALIZATION = "{initialization}"
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
USE_AMP = True
USE_BOTH_GPUS = True
RUN_TRAINING = False  # Set to True in your fork after enabling the Kaggle GPU.
RESUME_CHECKPOINT = None  # Example: "/kaggle/input/previous-run/last_checkpoint.pt"
OUTPUT_DIR = Path("/kaggle/working") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_MANIFEST_SHA256 = "{EXPECTED_MANIFEST_SHA}"
CLASS_NAMES = {EXPECTED_CLASSES!r}
NUM_CLASSES = len(CLASS_NAMES)
"""),
        md("""
## 2. Record the environment and make the run reproducible

Set Python, NumPy, and PyTorch seeds before creating the data loaders or model. Record the actual accelerator and library versions so results can be interpreted and resumed later.
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
print({
    "experiment": EXPERIMENT_ID,
    "python_seed": SEED,
    "device": str(DEVICE),
    "gpu_count": GPU_COUNT,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)],
    "torch": torch.__version__,
    "torchvision": __import__("torchvision").__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
})
"""),
        md("""
## 3. Locate and verify the published ROI dataset

The Kaggle input contains the already prepared 224 × 224 crops and a manifest. Verify the manifest checksum against the preparation receipt, check every referenced crop exists, validate the three class IDs, and confirm that parent groups do not cross the locked train/validation split. This dataset contains no test crops, and this notebook never opens the separate held-out test images.
"""),
        code("""
candidate_manifests = list(Path("/kaggle/input").glob("**/roi/manifest.csv"))
if not candidate_manifests:
    candidate_manifests = list(Path("/kaggle/input").glob("**/manifest.csv"))
if not candidate_manifests:
    raise FileNotFoundError("Attach trihuynhviprovcl/dentalselfie-gum-roi-v1, version 1, as a notebook input.")

MANIFEST_PATH = candidate_manifests[0]
ROI_ROOT = MANIFEST_PATH.parent
RECEIPT_PATH = ROI_ROOT / "receipt.json"
if not RECEIPT_PATH.is_file():
    raise FileNotFoundError(f"ROI receipt is missing beside {MANIFEST_PATH}")

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with open(RECEIPT_PATH, encoding="utf-8") as stream:
    ROI_RECEIPT = json.load(stream)
actual_manifest_sha = sha256_file(MANIFEST_PATH)
assert actual_manifest_sha == EXPECTED_MANIFEST_SHA256, "ROI manifest differs from the reviewed dataset version."
assert ROI_RECEIPT["manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "Receipt and reviewed manifest hash disagree."
assert ROI_RECEIPT["test_images_opened"] == 0, "Unexpected test-image processing in ROI receipt."

manifest = pd.read_csv(MANIFEST_PATH)
required_columns = {"crop_id", "parent_id", "group_id", "split", "class_id", "class_name", "local_path", "context_path"}
assert required_columns.issubset(manifest.columns), f"Manifest columns missing: {required_columns - set(manifest.columns)}"
assert set(manifest["split"].unique()) == {"train", "val"}, "Only locked train/validation ROI splits are expected."
assert set(manifest["class_id"].unique()).issubset({0, 1, 2}), "Unexpected class ID."
assert not manifest["crop_id"].duplicated().any(), "Duplicate crop identifiers found."
assert manifest.groupby("group_id")["split"].nunique().max() == 1, "A source group crosses train and validation."
for column in ("local_path", "context_path"):
    missing = [relative for relative in manifest[column] if not (ROI_ROOT / relative).is_file()]
    assert not missing, f"Missing files for {column}: {missing[:5]}"

print({"roi_root": str(ROI_ROOT), "manifest_sha256": actual_manifest_sha,
       "rows": len(manifest), "parents": manifest.parent_id.nunique(),
       "splits": manifest.split.value_counts().to_dict(),
       "receipt_crops": ROI_RECEIPT["crops"], "test_images_opened": ROI_RECEIPT["test_images_opened"]})
"""),
        md("""
## 4. Inspect the class distribution

Count ROIs rather than source photographs because each labeled box is one classification example. The no-visible-inflammation group is much smaller than the other classes. This screening baseline uses ordinary cross-entropy without class weights; class balancing must be studied as a separate controlled experiment.
"""),
        code("""
class_table = (manifest.groupby(["split", "class_id", "class_name"]).size()
               .rename("roi_count").reset_index())
display(class_table.pivot(index="class_name", columns="split", values="roi_count").fillna(0).astype(int))
class_table.pivot(index="class_name", columns="split", values="roi_count").plot(kind="bar", figsize=(9, 4))
plt.title("Gingival ROI class distribution")
plt.ylabel("Number of ROIs")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()
"""),
        md("""
## 5. Inspect original ROI crops and native box sizes

Display representative examples from each class and report the original box dimensions recorded before resizing. Upscaling a small crop standardizes the tensor shape but cannot restore details that were absent in the source image.
"""),
        code("""
fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(12, 3 * NUM_CLASSES))
for class_id, class_name in enumerate(CLASS_NAMES):
    examples = manifest[(manifest.split == "train") & (manifest.class_id == class_id)].head(4)
    for column, (_, row) in enumerate(examples.iterrows()):
        image = Image.open(ROI_ROOT / row.local_path).convert("RGB")
        axes[class_id, column].imshow(image)
        axes[class_id, column].set_title(f"{class_name}\\nbox {row.box_width:.0f} × {row.box_height:.0f}px")
        axes[class_id, column].axis("off")
for ax in axes.flat:
    if not ax.has_data():
        ax.axis("off")
plt.tight_layout()
plt.show()

display(manifest.groupby("split")[["box_width", "box_height"]].quantile([0.1, 0.5, 0.9]).round(1))
"""),
        md("""
## 6. Resize each crop to the model input size

The prepared ROI images are already 224 × 224 pixels. Apply an explicit resize in the model pipeline so the input rule is visible and robust to other image sizes. Bilinear interpolation is a practical standard for continuous-valued photographs.
"""),
        code("""
resize_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)
sample_row = manifest[manifest.split == "train"].iloc[0]
sample_image = Image.open(ROI_ROOT / sample_row.local_path).convert("RGB")
resized_image = resize_transform(sample_image)
print({"before": sample_image.size, "after": resized_image.size, "interpolation": "bilinear"})
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(sample_image); axes[0].set_title(f"Before: {sample_image.size}"); axes[0].axis("off")
axes[1].imshow(resized_image); axes[1].set_title(f"After: {resized_image.size}"); axes[1].axis("off")
plt.tight_layout(); plt.show()
"""),
        md("""
## 7. Define training-only augmentation

Small horizontal flips, rotations, and color changes model plausible framing and lighting variation. They are applied to training images only. Validation images receive deterministic resize and normalization so model comparisons use the same untouched examples.
"""),
        code("""
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7, interpolation=transforms.InterpolationMode.BILINEAR, fill=(128, 128, 128)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
])
validation_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for index, ax in enumerate(axes):
    ax.imshow(train_transform(sample_image)); ax.set_title("Original" if index == 0 else f"Train view {index}"); ax.axis("off")
plt.suptitle("Training augmentation examples")
plt.tight_layout(); plt.show()
"""),
        md("""
## 8. Convert pixels to tensors and normalize

`ToTensor` converts RGB pixels from integer values in `[0, 255]` to floating point values in `[0, 1]` with channel-first shape. ImageNet channel means and standard deviations then normalize both scratch and pretrained experiments identically. Using the same normalization keeps preprocessing from becoming an extra experimental factor.
"""),
        code("""
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
train_transform = transforms.Compose([train_transform, transforms.ToTensor(), normalize])
validation_transform = transforms.Compose([validation_transform, transforms.ToTensor(), normalize])

tensor_sample = transforms.ToTensor()(resized_image)
normalized_sample = normalize(tensor_sample)
print({"tensor_shape_C_H_W": tuple(tensor_sample.shape),
       "raw_tensor_min_max": (float(tensor_sample.min()), float(tensor_sample.max())),
       "normalized_min_max": (float(normalized_sample.min()), float(normalized_sample.max())),
       "mean": IMAGE_MEAN, "std": IMAGE_STD})
"""),
        md("""
## 9. Build the dataset and data loaders

The classifier reads the `local_path` crop and its `class_id`. The context crop is reserved for the later fusion study. Validation uses deterministic transforms. A source image can contribute several labeled ROIs, so metrics describe ROI examples and do not imply independent patients.
"""),
        code("""
class GingivalROIDataset(Dataset):
    def __init__(self, frame, root, transform):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.local_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        return image, int(row.class_id), str(row.crop_id)

train_frame = manifest[manifest.split == "train"].copy()
val_frame = manifest[manifest.split == "val"].copy()
train_dataset = GingivalROIDataset(train_frame, ROI_ROOT, train_transform)
val_dataset = GingivalROIDataset(val_frame, ROI_ROOT, validation_transform)
loader_options = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
                  "pin_memory": DEVICE.type == "cuda", "persistent_workers": NUM_WORKERS > 0}
loader_generator = torch.Generator().manual_seed(SEED)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, generator=loader_generator,
                          worker_init_fn=seed_worker, **loader_options)
val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                        worker_init_fn=seed_worker, **loader_options)
batch_images, batch_labels, batch_ids = next(iter(train_loader))
print({"train_examples": len(train_dataset), "validation_examples": len(val_dataset),
       "batch_images": tuple(batch_images.shape), "batch_labels": tuple(batch_labels.shape),
       "first_crop_ids": list(batch_ids[:5])})
"""),
    ]
    return cells


def model_cells(model_id: str):
    if model_id == "6.1":
        return [
            md("""
## 10. Build the custom CNN from basic layers

The baseline uses four convolution blocks. Each block extracts local patterns, normalizes activations, applies ReLU, and downsamples with max pooling. Adaptive average pooling produces a fixed-length vector before the three-class head. This compact model provides a from-scratch reference without external pretrained weights.
"""),
            code("""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
    def forward(self, x):
        return self.layers(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        channels = (3, 32, 64, 128, 256)
        self.features = nn.Sequential(*[
            ConvBlock(channels[index], channels[index + 1]) for index in range(len(channels) - 1)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

model = CustomCNN(NUM_CLASSES)
"""),
            md("""
## 11. Check the CNN structure and parameter count

Run one batch through the network before training. The output must contain one logit per class for each input image. This catches shape mistakes early and records the model size for the later resource comparison.
"""),
        ]
    return [
        md("""
## 10. Implement the ResNet18 residual block

Each basic block applies two 3 × 3 convolutions and adds the original input through a shortcut. When the channel count or resolution changes, a 1 × 1 projection makes the shortcut shape match. The identity path helps gradients pass through a deeper network.
"""),
        code("""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)
"""),
        md("""
## 11. Assemble the ResNet18 stages

The network has four residual stages with block counts `[2, 2, 2, 2]`. Its module names and tensor layout follow the standard ResNet18 structure. Notebook 6.2 trains this implementation from random initialization; notebook 6.3 loads matching ImageNet weights into the same structure.
"""),
        code("""
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    def _make_layer(self, channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels
        layers.extend(BasicBlock(self.in_channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

model = ResNet18(num_classes=NUM_CLASSES)
"""),
        md("""
## 12. Initialize the selected ResNet18 weights

This notebook uses the **same manually implemented ResNet18** as the scratch control. The pretrained variant loads the official torchvision `IMAGENET1K_V1` weights with a strict key check, then replaces only the 1,000-class ImageNet output layer with a three-class head. Kaggle Internet access is required the first time the weights are downloaded.
"""),
        code("""
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
reference = ResNet18(num_classes=1000)
if RUN_TRAINING:
    state_dict = weights.get_state_dict(progress=True, check_hash=True)
    load_result = reference.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    print({"initialization": "ImageNet IMAGENET1K_V1", "weights_url": weights.url,
           "strict_load": True, "classifier": "new 512-to-3 linear layer"})
else:
    print("Pretrained weights will be loaded when RUN_TRAINING=True.")
reference.fc = nn.Linear(512, NUM_CLASSES)
model = reference
"""),
        md("""
## 13. Verify the ResNet18 forward pass

The 224 × 224 image passes through the stem, four residual stages, global average pooling, and the three-class head. Record the output shape and parameter count before training.
"""),
    ]


def common_training_cells(model_id: str):
    return [
        md("""
## Check model output shape and parameter count

The model must return logits with shape `[batch_size, 3]`. If two T4 GPUs are attached, `DataParallel` distributes the batch across both; otherwise the notebook uses the available device. The reported parameter count is for one model replica.
"""),
        code("""
model = model.to(DEVICE)
if USE_BOTH_GPUS and GPU_COUNT > 1:
    model = nn.DataParallel(model)
model_for_count = model.module if isinstance(model, nn.DataParallel) else model
with torch.no_grad():
    test_logits = model(batch_images[:2].to(DEVICE))
assert tuple(test_logits.shape) == (2, NUM_CLASSES), tuple(test_logits.shape)
parameter_count = sum(parameter.numel() for parameter in model_for_count.parameters())
trainable_count = sum(parameter.numel() for parameter in model_for_count.parameters() if parameter.requires_grad)
print({"input_shape": tuple(batch_images[:2].shape), "logit_shape": tuple(test_logits.shape),
       "parameters": parameter_count, "trainable_parameters": trainable_count,
       "parallel_devices": GPU_COUNT if isinstance(model, nn.DataParallel) else 1})
"""),
        md("""
## Define the loss and optimization rule

Cross-entropy compares the three class logits with the integer class label. AdamW applies decoupled weight decay. The same declared learning rate and loss are used across these baseline notebooks; class weighting is left for a separate ablation because the classes are imbalanced.
"""),
        code("""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
print({"loss": "cross_entropy", "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
       "weight_decay": WEIGHT_DECAY, "scheduler": "ReduceLROnPlateau on validation macro-F1"})
"""),
        md("""
## Implement one training and validation pass

Mixed precision reduces activation memory and can accelerate T4 training. Validation runs without gradients or random augmentation. Each epoch records loss, accuracy, macro-F1, balanced accuracy, per-class recall, and the confusion matrix.
"""),
        code("""
def classification_metrics(y_true, y_pred, class_count=NUM_CLASSES):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    per_class_recall = np.divide(np.diag(matrix), true_count, out=np.zeros(class_count, dtype=float), where=true_count > 0)
    per_class_precision = np.divide(np.diag(matrix), predicted_count, out=np.zeros(class_count, dtype=float), where=predicted_count > 0)
    per_class_f1 = np.divide(2 * per_class_precision * per_class_recall,
                             per_class_precision + per_class_recall,
                             out=np.zeros(class_count, dtype=float),
                             where=(per_class_precision + per_class_recall) > 0)
    return {"confusion_matrix": matrix.tolist(), "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
            "macro_precision": float(per_class_precision.mean()), "macro_recall": float(per_class_recall.mean()),
            "macro_f1": float(per_class_f1.mean()), "balanced_accuracy": float(per_class_recall.mean()),
            "per_class_precision": per_class_precision.tolist(), "per_class_recall": per_class_recall.tolist(),
            "per_class_f1": per_class_f1.tolist()}

def run_epoch(model, loader, training):
    model.train(training)
    total_loss, total_examples = 0.0, 0
    all_targets, all_predictions = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = labels.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = classification_metrics(all_targets, all_predictions)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics
"""),
        md("""
## Save resumable checkpoints

Save the latest state every epoch and the best validation macro-F1 state separately. The checkpoint includes model, optimizer, scheduler, AMP scaler, random number generator states, epoch, configuration, and history. Data-loader worker state is not serialized, so a resumed run is not bitwise identical.
"""),
        code("""
def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_checkpoint(path, epoch, best_macro_f1, stale_epochs, history):
    payload = {"experiment_id": EXPERIMENT_ID, "epoch": epoch, "best_macro_f1": best_macro_f1,
        "stale_epochs": stale_epochs, "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history, "seed": SEED, "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "config": {"image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                   "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                   "mean": IMAGE_MEAN, "std": IMAGE_STD, "model": MODEL_TITLE}}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

def load_checkpoint(path):
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    assert payload["experiment_id"] == EXPERIMENT_ID, "Checkpoint belongs to another experiment."
    assert payload["manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "Checkpoint uses a different ROI manifest."
    unwrap_model(model).load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload
"""),
        md("""
## Train, validate, and resume

Training is disabled in the published notebook so publishing it does not consume GPU time. In your fork, enable a Kaggle GPU and set `RUN_TRAINING = True` in the configuration cell. The training loop saves a checkpoint each epoch, selects by validation macro-F1, and stops after the declared patience. Set `RESUME_CHECKPOINT` to an attached checkpoint to continue.
"""),
        code("""
history = []
start_epoch = 1
best_macro_f1 = -1.0
stale_epochs = 0
if RESUME_CHECKPOINT:
    resumed = load_checkpoint(RESUME_CHECKPOINT)
    previous_best = Path(RESUME_CHECKPOINT).with_name("best_checkpoint.pt")
    if previous_best.is_file():
        shutil.copy2(previous_best, OUTPUT_DIR / "best_checkpoint.pt")
    else:
        raise FileNotFoundError("Attach the complete previous output containing both best_checkpoint.pt and last_checkpoint.pt.")
    start_epoch = int(resumed["epoch"]) + 1
    best_macro_f1 = float(resumed["best_macro_f1"])
    stale_epochs = int(resumed["stale_epochs"])
    history = resumed["history"]
    print(f"Resumed after epoch {start_epoch - 1}; best validation macro-F1={best_macro_f1:.4f}")

if RUN_TRAINING:
    if DEVICE.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU before setting RUN_TRAINING=True.")
    started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, training=True)
        val_metrics = run_epoch(model, val_loader, training=False)
        scheduler.step(val_metrics["macro_f1"])
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                  **{f"train_{key}": value for key, value in train_metrics.items() if key != "confusion_matrix"},
                  **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(record)
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            save_checkpoint(OUTPUT_DIR / "best_checkpoint.pt", epoch, best_macro_f1, stale_epochs, history)
        else:
            stale_epochs += 1
        save_checkpoint(OUTPUT_DIR / "last_checkpoint.pt", epoch, best_macro_f1, stale_epochs, history)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | val macro-F1 {val_metrics['macro_f1']:.4f} | "
              f"val balanced accuracy {val_metrics['balanced_accuracy']:.4f} | "
              f"seconds {time.time() - started:.1f}")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation macro-F1 did not improve.")
            break
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "model": MODEL_TITLE, "initialization": INITIALIZATION,
        "seed": SEED, "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "epochs_completed": len(history), "best_validation_macro_f1": best_macro_f1,
        "elapsed_seconds": time.time() - started, "device": str(DEVICE),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)],
        "output_dir": str(OUTPUT_DIR)}, indent=2), encoding="utf-8")
    print("Saved checkpoints and run outputs to", OUTPUT_DIR)
else:
    print("Training skipped. Set RUN_TRAINING=True in the configuration cell of your GPU-enabled fork.")
"""),
        md("""
## Plot the saved learning curves

Plot training and validation loss and validation macro-F1. These curves show convergence and overfitting behavior; the selected checkpoint is still chosen using validation macro-F1 only.
"""),
        code("""
history_path = OUTPUT_DIR / "training_history.csv"
if history_path.is_file():
    history_frame = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_frame.epoch, history_frame.train_loss, label="Train")
    axes[0].plot(history_frame.epoch, history_frame.val_loss, label="Validation")
    axes[0].set_title("Cross-entropy loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history_frame.epoch, history_frame.val_macro_f1, marker="o")
    axes[1].set_title("Validation macro-F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()
else:
    print("No training history exists yet; run the training cell on a GPU to create it.")
"""),
        md("""
## Evaluate the best validation checkpoint

Reload the best checkpoint and report class-wise precision, recall, F1, and the confusion matrix. Validation is used for model selection; final test evaluation belongs to a later frozen-protocol stage.
"""),
        code("""
best_path = OUTPUT_DIR / "best_checkpoint.pt"
if best_path.is_file():
    best_payload = torch.load(best_path, map_location=DEVICE, weights_only=False)
    unwrap_model(model).load_state_dict(best_payload["model_state"], strict=True)
    best_validation = run_epoch(model, val_loader, training=False)
    report = {"experiment_id": EXPERIMENT_ID, "selected_epoch": best_payload["epoch"],
              "selection_metric": "validation macro-F1", "validation": best_validation,
              "class_names": CLASS_NAMES, "manifest_sha256": EXPECTED_MANIFEST_SHA256}
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
else:
    print("Best checkpoint not found. Run the training cell first.")
"""),
        md("""
## Inspect correct and incorrect validation predictions

Show representative examples with the true and predicted class. Errors help explain which appearance groups overlap. They are ROI-level observations, not clinical conclusions.
"""),
        code("""
if (OUTPUT_DIR / "best_checkpoint.pt").is_file():
    model.eval()
    errors, correct = [], []
    with torch.no_grad():
        for images, labels, crop_ids in val_loader:
            logits = model(images.to(DEVICE))
            predictions = logits.argmax(dim=1).cpu().tolist()
            for image_tensor, label, prediction, crop_id in zip(images, labels.tolist(), predictions, crop_ids):
                item = (image_tensor, label, prediction, crop_id)
                (correct if label == prediction else errors).append(item)
    selected = errors[:6] + correct[:6]
    if selected:
        fig, axes = plt.subplots(2, 6, figsize=(15, 5))
        for ax, (image_tensor, label, prediction, crop_id) in zip(axes.flat, selected):
            image = image_tensor.permute(1, 2, 0).numpy() * np.asarray(IMAGE_STD) + np.asarray(IMAGE_MEAN)
            ax.imshow(np.clip(image, 0, 1))
            ax.set_title(f"True: {CLASS_NAMES[label]}\\nPred: {CLASS_NAMES[prediction]}\\n{crop_id[:12]}", fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(selected):]: ax.axis("off")
        plt.tight_layout(); plt.show()
    print({"incorrect_examples_shown": min(6, len(errors)), "correct_examples_shown": min(6, len(correct))})
else:
    print("Run training before reviewing predictions.")
"""),
        md("""
## Research interpretation and output checklist

Record whether the model converged, which class recalls remain weak, and how its parameter count and runtime compare with the other two baselines. This notebook addresses classification of already localized gingival ROIs. It does not measure detector recall, end-to-end screening performance, patient-level diagnosis, or mobile-device latency.

Expected files in `/kaggle/working/{experiment_id}/`: `best_checkpoint.pt`, `last_checkpoint.pt`, `training_history.csv`, `run_metadata.json`, and `validation_report.json`. Preserve the Kaggle output artifact after the run so another session can attach it for resumption.
"""),
    ]


def build_notebook(model_id: str, slug: str, title: str, initialization: str):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = shared_cells(model_id, title, initialization) + model_cells(model_id) + common_training_cells(model_id)
    title = ascii_dashes(title)
    for cell in notebook.cells:
        cell.source = ascii_dashes(cell.source)
    kaggle_slug = {
        "dentalselfie-06-1-cnn": "dentalselfie-6-1-custom-cnn-baseline",
        "dentalselfie-06-2-resnet18-scratch": "dentalselfie-6-2-resnet18-from-scratch",
        "dentalselfie-06-3-resnet18-pretrained": "dentalselfie-6-3-imagenet-pretrained-resnet18",
    }[slug]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"id": f"trihuynhviprovcl/{kaggle_slug}", "title": title,
                   "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
                   "is_private": False, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": ["trihuynhviprovcl/dentalselfie-gum-roi-v1/1"],
                   "competition_sources": [], "kernel_sources": []},
    }
    path = NOTEBOOKS / f"{slug}.ipynb"
    nbformat.write(notebook, path)
    return path


def main():
    experiments = [
        ("6.1", "dentalselfie-06-1-cnn", "DentalSelfie 6.1 - Custom CNN Baseline", "Random initialization"),
        ("6.2", "dentalselfie-06-2-resnet18-scratch", "DentalSelfie 6.2 - ResNet18 from Scratch", "Random initialization"),
        ("6.3", "dentalselfie-06-3-resnet18-pretrained", "DentalSelfie 6.3 - ImageNet Pretrained ResNet18", "Torchvision IMAGENET1K_V1"),
    ]
    for experiment in experiments:
        path = build_notebook(*experiment)
        print(path)


if __name__ == "__main__":
    main()
