"""
Training Step 1: Head-Only ResNet-18 Training
---------------------------------------------------
Train a 2-class overpronation/normal classifier (frozen ResNet-18 head-only).

- Use prepared and augmented stance leg/foot (cropped) images.
- Group images to avoid leakage between train/val.
- Stratify by group-level labels (from prepared images) to keep class balance.
- Apply weighted loss if class imbalance > 60/40.
- Freeze backbone and BatchNorm stats.
- Save the model once train AUC > threshold, and on Val AUC improvement.
- Early stopping when patience is reached.
- Save logs to the phase folder.

Outputs (example for RUN_NAME="headonly"):
  data/training/phase1_train_headonly/models/best_resnet18_headonly.pth
  data/training/phase1_train_headonly/logs/training_log_headonly.csv
"""

import os, re, csv, random, shutil
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from tqdm import tqdm
from PIL import Image

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_ROOT = os.path.join(DATA_DIR, "training")
PHASE_DIR = os.path.join(TRAIN_ROOT, "phase1_train_headonly")
DATA_PREPARED = os.path.join(DATA_DIR, "prepared")
DATA_AUGMENTED = os.path.join(DATA_DIR, "augmented")

# ===== SETTINGS =====
RUN_NAME = "headonly"           # suffix for saved files
CLEAN_TRAINING = True           # remove previous outputs before training

BATCH_SIZE = 4
NUM_EPOCHS = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-5
PATIENCE = 8
DROPOUT_RATE = 0.4
TRAIN_RATIO = 0.6
TRAIN_AUC_THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def set_seed(seed=42):
    """Ensure deterministic behavior across runs."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def clean_training_dir(phase_dir):
    """Clear existing phase folder and recreate subfolders /models and /logs."""
    if os.path.exists(phase_dir):
        for fname in os.listdir(phase_dir):
            fpath = os.path.join(phase_dir, fname)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
    os.makedirs(os.path.join(phase_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, "logs"), exist_ok=True)

def base_id(path):
    """Return numeric prefix from filename, used for grouping related images."""
    fname = os.path.basename(path)
    m = re.match(r"(\d+)", fname)
    return m.group(1) if m else fname

def make_transforms():
    """Return grayscale→RGB transforms normalized by ImageNet RGB mean/std."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])

class PathDataset(Dataset):
    """Custom dataset that loads arbitrary file paths after manual split."""
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]

def build_dataloaders(data_prepared, data_augmented, transform, train_ratio, batch_size):
    """Perform a grouped stratified train/val split by group labels."""
    ds_prepared = datasets.ImageFolder(root=data_prepared, transform=transform)
    ds_augmented = datasets.ImageFolder(root=data_augmented, transform=transform)
    assert ds_prepared.classes == ds_augmented.classes, "Class mismatch!"
    print(f"Classes: {ds_prepared.classes}")

    # --- Collect paths + labels + group ids ---
    all_paths, all_labels, all_groups = [], [], []
    for ds in [ds_prepared, ds_augmented]:
        for p, y in ds.samples:
            all_paths.append(p); all_labels.append(y); all_groups.append(base_id(p))
    all_paths, all_labels, all_groups = np.array(all_paths), np.array(all_labels), np.array(all_groups)

    # --- Map each group to its class (based on prepared image) ---
    group_to_label = {}
    for p, y, g in zip(all_paths, all_labels, all_groups):
        if p.startswith(data_prepared):
            group_to_label[g] = y
    unique_groups = np.unique(all_groups)
    group_labels = np.array([group_to_label[g] for g in unique_groups])

    # --- Stratified split at group level ---
    test_size = 1.0 - train_ratio
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    (train_g_idx, val_g_idx), = sss.split(unique_groups, group_labels)
    train_groups = unique_groups[train_g_idx]
    val_groups   = unique_groups[val_g_idx]
    assert len(set(train_groups) & set(val_groups)) == 0, "Overlap between train/val groups!"

    print(f"Grouped stratified split (by prepared labels): Train={len(train_groups)} groups, "
          f"Val={len(val_groups)} groups (~{int(train_ratio*100)}/{int((1-train_ratio)*100)})")

    # --- Build masks ---
    train_mask = np.isin(all_groups, train_groups)
    val_mask   = np.isin(all_groups, val_groups)

    train_paths, train_labels = all_paths[train_mask], all_labels[train_mask]
    val_paths, val_labels     = all_paths[val_mask],   all_labels[val_mask]

    # --- Filter validation to prepared-only ---
    val_paths  = np.array([p for p in val_paths if p.startswith(data_prepared)])
    val_labels = np.array([all_labels[np.where(all_paths==p)[0][0]] for p in val_paths])

    # --- Check class balance ---
    unique, counts = np.unique(val_labels, return_counts=True)
    print("Validation label counts:", dict(zip(unique, counts)))

    # --- Build dataloaders ---
    train_loader = DataLoader(PathDataset(train_paths, train_labels, transform), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PathDataset(val_paths, val_labels, transform), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_paths, train_labels

def build_model(dropout_rate, learning_rate, weight_decay, device):
    """Initialize a frozen ResNet-18 (pretrained) with a new head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval(); m.track_running_stats = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(num_features, 2))
    model = model.to(device)
    optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, optimizer

def get_loss(train_labels, device):
    """Return CrossEntropyLoss, weighted if imbalance > 60/40."""
    counts = np.bincount(train_labels)
    ratio = counts / counts.sum()
    if np.max(ratio) > 0.6:
        weights = counts.sum() / (2 * counts)
        print("Class imbalance > 60/40, applying weighted loss.")
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    return nn.CrossEntropyLoss()

# ------------------------------------------------------------
# Main Training Logic
# ------------------------------------------------------------

def main():
    """Execute the head-only training phase."""
    set_seed(42)
    if CLEAN_TRAINING:
        clean_training_dir(PHASE_DIR)
    else:
        os.makedirs(os.path.join(PHASE_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(PHASE_DIR, "logs"), exist_ok=True)

    SAVE_MODEL_PATH = os.path.join(PHASE_DIR, "models", f"best_resnet18_{RUN_NAME}.pth")
    LOG_FILE = os.path.join(PHASE_DIR, "logs", f"training_log_{RUN_NAME}.csv")

    transform = make_transforms()
    train_loader, val_loader, train_paths, train_labels = build_dataloaders(
        DATA_PREPARED, DATA_AUGMENTED, transform, TRAIN_RATIO, BATCH_SIZE
    )
    criterion = get_loss(train_labels, DEVICE)
    model, optimizer = build_model(DROPOUT_RATE, LEARNING_RATE, WEIGHT_DECAY, DEVICE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy="cos"
    )

    best_auc, bad_epochs, model_saved = 0.0, 0, False
    tracking_started, tracking_epoch = False, None
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_auc",
                                "train_auc_originals", "val_loss", "val_acc", "val_auc"])

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        probs_train, labels_train = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.fc.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs_train.extend(torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy())
            labels_train.extend(labels.cpu().numpy())

        train_loss /= total
        train_acc = correct / total
        try:
            train_auc = roc_auc_score(labels_train, probs_train)
        except:
            train_auc = float("nan")

        # === VALIDATION ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        probs_val, labels_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                probs_val.extend(torch.softmax(outputs, dim=1)[:,1].cpu().numpy())
                labels_val.extend(labels.cpu().numpy())
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total
        try:
            val_auc = roc_auc_score(labels_val, probs_val)
        except:
            val_auc = float("nan")

        # === ORIGINALS-ONLY TRAIN AUC ===
        orig_paths = [p for p in train_paths if p.startswith(DATA_PREPARED)]
        orig_labels = [train_labels[i] for i, p in enumerate(train_paths) if p.startswith(DATA_PREPARED)]
        if len(orig_paths) > 0:
            probs_orig, labels_orig = [], []
            with torch.no_grad():
                for i in range(0, len(orig_paths), BATCH_SIZE):
                    imgs_batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in orig_paths[i:i+BATCH_SIZE]]).to(DEVICE)
                    outputs = model(imgs_batch)
                    probs_orig.extend(torch.softmax(outputs, dim=1)[:,1].cpu().numpy())
                    labels_orig.extend(orig_labels[i:i+BATCH_SIZE])
            try:
                train_orig_auc = roc_auc_score(labels_orig, probs_orig)
            except:
                train_orig_auc = float("nan")
            print(f"TrainAUC(originals-only)={train_orig_auc:.3f}")
        else:
            train_orig_auc = float("nan")

        # === LOGGING ===
        print(f"Epoch {epoch+1:02d}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, TrainAUC={train_auc:.3f} | ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, ValAUC={val_auc:.3f}")
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, train_loss, train_acc, train_auc, train_orig_auc, val_loss, val_acc, val_auc])

        # === MODEL SAVING LOGIC ===
        if epoch == 0:
            print("Epoch 1 skipped (warm-up).")
            continue
        if not tracking_started and train_auc > TRAIN_AUC_THRESHOLD:
            tracking_started, tracking_epoch = True, epoch + 1
            best_auc, bad_epochs = val_auc, 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            model_saved = True
            print(f"Tracking started (epoch {tracking_epoch}, train_auc={train_auc:.3f}, val_auc={val_auc:.3f})")
            continue
        if tracking_started:
            if val_auc > best_auc + 1e-4 and train_auc > TRAIN_AUC_THRESHOLD:
                best_auc, bad_epochs = val_auc, 0
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                model_saved = True
                print(f"New best model saved (ValAUC={best_auc:.3f}).")
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
        else:
            print("Still warming up — train_auc below threshold.")

    print(f"\nTraining complete. Best validation AUC: {best_auc:.3f}")
    if tracking_epoch:
        print(f"Tracking began at epoch {tracking_epoch}.")
    if model_saved:
        print(f"Model saved to: {SAVE_MODEL_PATH}")
    else:
        print("No model met saving criteria. Consider adjusting thresholds or epochs.")


if __name__ == "__main__":
    main()
