"""
Training Step 2: Fine-Tune ResNet-18
---------------------------------------------------
Fine-tune a 2-class classifier with grouped 5-fold cross-validation.

- Load best model from Phase 1.
- Unfreeze one of three options of backbone layers (layer4/layer4.1/
  layer4_lastconv) after warm-up.
- Keep BatchNorm frozen.
- Uses grouped and stratified 5-fold cross-validation by group.
- Validate with prepared-only images (no augmentations).
- Apply weighted CrossEntropyLoss loss with label smoothing if class imbalance
  > 60/40.
- Starts OneCycleLR after HEAD_WARMUP_EPOCHS for stable warm-up.
- Save the model once train AUC > threshold, and on Val AUC improvement.
- Early stopping when patience is reached.
- Save logs to the phase folder.

Outputs (example for fold0):
  data/training/phase2_finetune_resnet18/models/best_resnet18_finetune_fold0.pth
  data/training/phase2_finetune_resnet18/logs/training_log_finetune_fold0.csv
  data/training/phase2_finetune_resnet18/summary.csv
"""

import os, re, csv, random, statistics, shutil
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
from PIL import Image

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_ROOT = os.path.join(DATA_DIR, "training")
PHASE1_BEST = os.path.join(TRAIN_ROOT, "phase1_train_headonly", "models", "best_resnet18_headonly.pth")
PHASE_DIR = os.path.join(TRAIN_ROOT, "phase2_finetune_resnet18")
DATA_PREPARED = os.path.join(DATA_DIR, "prepared")
DATA_AUGMENTED = os.path.join(DATA_DIR, "augmented")

# ===== SETTINGS =====
RUN_NAME = "finetune_resnet18"
CLEAN_TRAINING = True

FOLDS = 5
BATCH_SIZE = 4
NUM_EPOCHS = 35
PATIENCE = 10
MIN_DELTA = 0.002
TRAIN_AUC_THRESHOLD = 0.7

LR_HEAD = 2e-4
LR_BACKBONE = 2e-5
WEIGHT_DECAY = 5e-5
DROPOUT_RATE = 0.5
LABEL_SMOOTH = 0.05
UNFREEZE_MODE = "4lastconv"   # 4 / 4.1 / 4lastconv
HEAD_WARMUP_EPOCHS = 2        # unfreeze after this

# OneCycleLR
PCT_START = 0.2
DIV_FACTOR = 10
FINAL_DIV_FACTOR = 5e3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def set_seed(seed=42):
    """Ensure deterministic behavior across runs."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def clean_training_dir(phase_dir):
    """Clear existing phase folder and recreate subfolders /models and /logs."""
    if os.path.exists(phase_dir):
        for f in os.listdir(phase_dir):
            path = os.path.join(phase_dir, f)
            if os.path.isfile(path) or os.path.islink(path): 
                os.remove(path)
            else: 
                shutil.rmtree(path)
    os.makedirs(os.path.join(phase_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, "logs"), exist_ok=True)

def base_id(path):
    """Return numeric prefix from filename, used for grouping related images."""
    fname = os.path.basename(path)
    m = re.match(r"(\d+)", fname)
    return m.group(1) if m else fname

def make_transforms():
    """Return grayscale→RGB transform normalized by ImageNet mean/std."""
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
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

def get_loss(train_labels):
    """Return CrossEntropyLoss, weighted + label smoothing if imbalance >60/40."""
    counts = np.bincount(train_labels)
    ratio = counts / counts.sum()
    if np.max(ratio) > 0.6:
        weights = counts.sum() / (2 * counts)
        print("Class imbalance > 60/40 → weighted CE + label smoothing.")
        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE),
            label_smoothing=LABEL_SMOOTH
        )
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

def unfreeze_backbone(model, mode):
    """Unfreeze selected backbone layers (layer4/layer4.1/layer4_lastconv) and keep BatchNorm frozen."""
    for n,p in model.named_parameters(): 
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval(); m.track_running_stats = False
    if mode == "4":
        for n,p in model.named_parameters():
            if "layer4" in n or "fc" in n: 
                p.requires_grad = True
        print("Unfrozen: layer4 (all) + fc")
    elif mode == "4.1":
        for n,p in model.named_parameters():
            if "layer4.1" in n or "fc" in n: 
                p.requires_grad = True
        print("Unfrozen: layer4.1 + fc")
    elif mode == "4lastconv":
        for n,p in model.named_parameters():
            if ("layer4.1.conv2" in n) or ("layer4.1.downsample.0" in n) or ("fc" in n):
                p.requires_grad = True
        print("Unfrozen: layer4_lastconv + fc")

def build_optimizer(model):
    """Build AdamW optimizer with separate learning rates for head and backbone."""
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and "fc" in n]
    bb_params   = [p for n,p in model.named_parameters() if p.requires_grad and "fc" not in n]
    return optim.AdamW([
        {"params": bb_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

# ------------------------------------------------------------
# Main Cross-Validation Training Logic
# ------------------------------------------------------------

def main():
    set_seed(SEED)
    if CLEAN_TRAINING: 
        clean_training_dir(PHASE_DIR)
    else:
        os.makedirs(os.path.join(PHASE_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(PHASE_DIR, "logs"), exist_ok=True)

    transform = make_transforms()

    ds_prepared  = datasets.ImageFolder(root=DATA_PREPARED,  transform=transform)
    ds_augmented = datasets.ImageFolder(root=DATA_AUGMENTED, transform=transform)
    assert ds_prepared.classes == ds_augmented.classes
    print(f"Classes: {ds_prepared.classes}")

    all_paths, all_labels, all_groups = [], [], []
    for ds in [ds_prepared, ds_augmented]:
        for p, y in ds.samples:
            all_paths.append(p); all_labels.append(y); all_groups.append(base_id(p))
    all_paths, all_labels, all_groups = np.array(all_paths), np.array(all_labels), np.array(all_groups)

    group_to_label = {base_id(p): y for p, y in ds_prepared.samples}
    unique_groups = np.unique(all_groups)
    group_labels = np.array([group_to_label[g] for g in unique_groups])

    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(unique_groups, group_labels, groups=unique_groups)):
        print(f"\n========== Fold {fold}/{FOLDS-1} ==========")
        train_groups, val_groups = unique_groups[train_idx], unique_groups[val_idx]
        train_mask, val_mask = np.isin(all_groups, train_groups), np.isin(all_groups, val_groups)
        train_paths, train_labels = all_paths[train_mask], all_labels[train_mask]
        val_paths, val_labels = all_paths[val_mask], all_labels[val_mask]
        val_paths = np.array([p for p in val_paths if p.startswith(DATA_PREPARED)])
        val_labels = np.array([all_labels[np.where(all_paths==p)[0][0]] for p in val_paths])

        train_loader = DataLoader(PathDataset(train_paths, train_labels, transform),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(PathDataset(val_paths, val_labels, transform),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval(); m.track_running_stats = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(num_features, 2))
        model = model.to(DEVICE)

        if os.path.exists(PHASE1_BEST):
            model.load_state_dict(torch.load(PHASE1_BEST, map_location=DEVICE), strict=False)
            print("Loaded Phase 1 baseline weights.")

        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
        optimizer = optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = None
        unfrozen = False

        criterion = get_loss(train_labels)

        LOG_FILE = os.path.join(PHASE_DIR, "logs", f"training_log_finetune_fold{fold}.csv")
        SAVE_PATH = os.path.join(PHASE_DIR, "models", f"best_resnet18_finetune_fold{fold}.pth")
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","train_acc","train_auc",
                                    "train_auc_originals","val_loss","val_acc","val_auc"])

        best_auc, bad_epochs = 0.0, 0
        tracking_started, tracking_epoch = False, None

        for epoch in range(NUM_EPOCHS):
            # Unfreeze after warm-up
            if (epoch + 1) == (HEAD_WARMUP_EPOCHS + 1) and not unfrozen:
                unfreeze_backbone(model, UNFREEZE_MODE)
                optimizer = build_optimizer(model)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[pg["lr"] for pg in optimizer.param_groups],
                    epochs=NUM_EPOCHS - epoch,
                    steps_per_epoch=len(train_loader),
                    pct_start=PCT_START,
                    div_factor=DIV_FACTOR,
                    final_div_factor=FINAL_DIV_FACTOR,
                    anneal_strategy="cos",
                    cycle_momentum=False
                )
                unfrozen = True
                print(f"Backbone unfrozen at epoch {epoch+1} → OneCycleLR started.")

            model.train()
            train_loss, correct, total = 0.0, 0, 0
            probs_train, labels_train = [], []

            for imgs, labels in tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
                train_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs_train.extend(torch.softmax(out,1)[:,1].detach().cpu().numpy())
                labels_train.extend(labels.cpu().numpy())

            train_loss /= total; train_acc = correct/total
            try: train_auc = roc_auc_score(labels_train, probs_train)
            except: train_auc = float("nan")

            # === ORIGINALS-ONLY TRAIN AUC (dropout OFF) === 
            orig_paths = [p for p in train_paths if p.startswith(DATA_PREPARED)]
            orig_labels = [train_labels[i] for i,p in enumerate(train_paths) if p.startswith(DATA_PREPARED)]
            if len(orig_paths) > 0:
                was_training = model.training; model.eval()
                probs_orig, labels_orig = [], []
                with torch.no_grad():
                    for i in range(0, len(orig_paths), BATCH_SIZE):
                        imgs_batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in orig_paths[i:i+BATCH_SIZE]]).to(DEVICE)
                        out = model(imgs_batch)
                        probs_orig.extend(torch.softmax(out,1)[:,1].cpu().numpy())
                        labels_orig.extend(orig_labels[i:i+BATCH_SIZE])
                try: train_orig_auc = roc_auc_score(labels_orig, probs_orig)
                except: train_orig_auc = float("nan")
                if was_training: model.train()
                print(f"TrainAUC(originals-only)={train_orig_auc:.3f}")
            else:
                train_orig_auc = float("nan")

            # === VALIDATION ===
            model.eval(); val_loss, vcorrect, vtotal = 0.0, 0, 0
            probs_val, labels_val = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    out = model(imgs)
                    loss = criterion(out, labels)
                    val_loss += loss.item() * imgs.size(0)
                    probs_val.extend(torch.softmax(out,1)[:,1].cpu().numpy())
                    preds = out.argmax(1)
                    vcorrect += (preds == labels).sum().item()
                    vtotal += labels.size(0)
                    labels_val.extend(labels.cpu().numpy())
            val_loss /= vtotal; val_acc = vcorrect/vtotal
            try: val_auc = roc_auc_score(labels_val, probs_val)
            except: val_auc = float("nan")

            # === LOGGING ===
            print(f"Epoch {epoch+1:02d}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
                  f"TrainAUC={train_auc:.3f} | ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, ValAUC={val_auc:.3f}")
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([epoch+1,train_loss,train_acc,train_auc,
                                        train_orig_auc,val_loss,val_acc,val_auc])

            # === MODEL SAVING LOGIC ===
            if epoch == 0:
                print("Epoch 1 skipped (warm-up).")
                continue

            if not tracking_started and unfrozen and train_auc > TRAIN_AUC_THRESHOLD:
                tracking_started, tracking_epoch = True, epoch + 1
                best_auc, bad_epochs = val_auc, 0
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Tracking started (epoch {tracking_epoch}, trainAUC={train_auc:.3f}, valAUC={val_auc:.3f})")
                continue

            if tracking_started and unfrozen:
                if val_auc > best_auc + MIN_DELTA and train_auc > TRAIN_AUC_THRESHOLD:
                    best_auc, bad_epochs = val_auc, 0
                    torch.save(model.state_dict(), SAVE_PATH)
                    print(f"New best model saved (ValAUC={best_auc:.3f})")
                else:
                    bad_epochs += 1
                    if bad_epochs >= PATIENCE:
                        print(f"⏹️ Early stopping at epoch {epoch+1}.")
                        break
            else:
                print("Still warming up — trainAUC below threshold or backbone not yet unfrozen.")

        print(f"🎯 Fold {fold} done | Best Val AUC = {best_auc:.3f}")
        fold_aucs.append(best_auc)

    mean_auc = statistics.mean(fold_aucs)
    std_auc  = statistics.stdev(fold_aucs)
    summary_path = os.path.join(PHASE_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold","best_val_auc"])
        for i,a in enumerate(fold_aucs): w.writerow([i,a])
        w.writerow([]); w.writerow(["mean",mean_auc]); w.writerow(["std",std_auc])

    print("\n===== Phase 2 complete =====")
    print(f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Summary → {summary_path}")
    print(f"Models / logs → {PHASE_DIR}")

if __name__ == "__main__":
    main()
