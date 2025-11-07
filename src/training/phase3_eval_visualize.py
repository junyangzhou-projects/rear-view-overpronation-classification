"""
 Training Step 3: Evaluate with Grad-CAM Visualization
 --------------------------------------------------
 Evaluation and summarization for Phase 2 ResNet-18 overpronation classifier.

 - Read logs from Phase 2.
 - Aggregate Val AUC/Acc (mean ± SD).
 - Generate Grad-CAM heatmap for prepared (stance leg/foot cropped) images.

 Outputs (example for fold0):
   data/training/phase3_results/metrics/fold0.csv
   data/training/phase3_results/metrics/aggregate_summary.csv
   data/training/phase3_results/visualizations/gradcam_fold0/*.jpg
"""


import os, csv, shutil, statistics, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_ROOT = os.path.join(DATA_DIR, "training")

PHASE2_DIR = os.path.join(TRAIN_ROOT, "phase2_finetune_resnet18")
PHASE3_DIR = os.path.join(TRAIN_ROOT, "phase3_eval_visualize")
DATA_PREPARED = os.path.join(DATA_DIR, "prepared")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===== SETTINGS =====
N_FOLDS = 5
VISUALIZE_SAMPLES = None   # None -> use all prepared images
CLEAN_PHASE_DIR = True

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def clean_phase_dir(phase_dir):
    """Clear existing phase folder and recreate subfolders."""
    if os.path.exists(phase_dir):
        for f in os.listdir(phase_dir):
            path = os.path.join(phase_dir, f)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
    os.makedirs(os.path.join(phase_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, "visualizations"), exist_ok=True)
    print(f"Cleaned Phase 3 directory → {phase_dir}")

def read_val_metrics(fold):
    """Read best epoch metrics (highest val_auc) from Phase 2 log."""
    log_file = os.path.join(PHASE2_DIR, "logs", f"training_log_finetune_fold{fold}.csv")
    with open(log_file, "r") as f:
        rows = list(csv.DictReader(f))
    # Select the epoch with the highest validation AUC
    best = max(rows, key=lambda r: float(r["val_auc"]))
    val_acc = float(best["val_acc"])
    val_auc = float(best["val_auc"])
    return val_acc, val_auc

# ------------------------------------------------------------
# Main Phase 3 Logic
# ------------------------------------------------------------

def main():
    if CLEAN_PHASE_DIR:
        clean_phase_dir(PHASE3_DIR)
    else:
        os.makedirs(os.path.join(PHASE3_DIR, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(PHASE3_DIR, "visualizations"), exist_ok=True)

    # === Aggregate per-fold metrics ===
    fold_accs, fold_aucs = [], []

    for fold in range(N_FOLDS):
        val_acc, val_auc = read_val_metrics(fold)
        fold_accs.append(val_acc)
        fold_aucs.append(val_auc)
        with open(os.path.join(PHASE3_DIR, "metrics", f"fold{fold}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "val_acc", "val_auc"])
            w.writerow([fold, val_acc, val_auc])

    mean_acc = statistics.mean(fold_accs)
    std_acc  = statistics.stdev(fold_accs)
    mean_auc = statistics.mean(fold_aucs)
    std_auc  = statistics.stdev(fold_aucs)

    summary_file = os.path.join(PHASE3_DIR, "metrics", "aggregate_summary.csv")
    with open(summary_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        w.writerow(["val_acc", mean_acc, std_acc])
        w.writerow(["val_auc", mean_auc, std_auc])

    print("===== Phase 3 Summary =====")
    print(f"Val Acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Val AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"→ Saved to {summary_file}")

    # === Grad-CAM Visualization ===
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        model_paths = [
            os.path.join(PHASE2_DIR, "models", f"best_resnet18_finetune_fold{f}.pth")
            for f in range(N_FOLDS)
        ]

        valid_exts = (".jpg", ".jpeg", ".png")
        all_imgs = []
        for root, _, files in os.walk(DATA_PREPARED):
            for f in sorted(files):
                if f.lower().endswith(valid_exts):
                    all_imgs.append(os.path.join(root, f))

        if VISUALIZE_SAMPLES:
            all_imgs = all_imgs[:VISUALIZE_SAMPLES]

        print(f"Found {len(all_imgs)} prepared images for Grad-CAM.")

        for fold, mpath in enumerate(model_paths):
            model = models.resnet18()
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
            model.load_state_dict(torch.load(mpath, map_location=DEVICE))
            model.to(DEVICE).eval()

            target_layer = model.layer4[-1]
            cam = GradCAM(model=model, target_layers=[target_layer])

            vis_dir = os.path.join(PHASE3_DIR, "visualizations", f"gradcam_fold{fold}")
            os.makedirs(vis_dir, exist_ok=True)

            for img_path in tqdm(all_imgs, desc=f"Grad-CAM fold{fold}"):
                if not os.path.isfile(img_path):
                    continue

                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(img_t)
                p_normal, p_over = F.softmax(output, dim=1)[0].cpu().numpy()

                # Generate Grad-CAM heatmap
                grayscale_cam = cam(input_tensor=img_t, targets=None, eigen_smooth=False)[0]

                # Overlay
                rgb = np.float32(img.resize((224, 224))) / 255.0
                overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

                true_label = os.path.basename(os.path.dirname(img_path)).capitalize()
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                h, w, _ = overlay_bgr.shape
                banner_h = int(h / 7.5)
                cv2.rectangle(overlay_bgr, (0, 0), (w, banner_h), (0, 0, 0), -1)

                font_scale = max(h, w) / 700.0
                thickness = 1 if h < 500 else 2
                y1 = int(banner_h * 0.42)
                y2 = int(banner_h * 0.84)

                cv2.putText(
                    overlay_bgr, f"Fold {fold} | True: {true_label}",
                    (8, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )
                
                cv2.putText(
                    overlay_bgr,
                    f"P(normal)={p_normal:.2f} | P(overpronation)={p_over:.2f}",
                    (8, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )

                out_name = os.path.basename(img_path).replace(".jpg", "_cam.jpg")
                out_path = os.path.join(vis_dir, out_name)
                cv2.imwrite(out_path, overlay_bgr)

        print("Grad-CAM visualizations complete.")
    except Exception as e:
        print("Grad-CAM skipped:", e)


if __name__ == "__main__":
    main()
