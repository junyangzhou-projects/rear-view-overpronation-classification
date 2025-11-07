"""
Experiment Step 1 — Pose-Angle (MediaPipe) vs. Manual Labels
---------------------------------------------------
- Read full-body photos in data/raw/ and stance side from 
  data/metadata/coords.json.
- Get MediaPipe Pose to estimate stance leg ankle angle (between knee->ankle
  and ankle->heel).
- Compare to the manual angles/labels from data/metadata/labels.csv.

- Save outputs to data/experiments/e1_pose_angle_experiment/:
    - results/e1_pose_results.csv with columns:
      filename, side, manual_angle, mediapipe_angle, error_deg, manual_label,
      mediapipe_label (for invalid detections: mediapipe_angle/mediapipe_label=
      "invalid", error_deg="").
    - visualizations folder saves the MidiaPipe pose and angle annotated with:
        1) Knee/ankle/heel joints (green if visibility≥0.5, else red)
        2) Lines showing knee→ankle and ankle→heel
        3) Text block at the top showing:
           Manual degree/label, Mediapipe degree/label
           Visibilities for the joints
    - invalid_visualizations folder saves the invalid detections.
    - results/summary.txt with timestamp, thresholds, and overall metrics.

INPUT:  data/raw/, data/metadata/labels.csv, data/metadata/coords.json
OUTPUT: data/experiments/e1_pose_angle_experiment/results/e1_pose_results.csv
        data/experiments/e1_pose_angle_experiment/results/summary.txt
        data/experiments/e1_pose_angle_experiment/visualizations/*.jpg
        data/experiments/e1_pose_angle_experiment/invalid_visualizations/*.jpg
"""

import os, re, cv2, json, math, shutil
import numpy as np, pandas as pd, mediapipe as mp
from datetime import datetime

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
META_DIR = os.path.join(ROOT_DIR, "data/metadata")
LABELS_PATH = os.path.join(META_DIR, "labels.csv")
COORDS_PATH = os.path.join(META_DIR, "coords.json")

EXP_ROOT = os.path.join(ROOT_DIR, "data/experiments/e1_pose_angle_experiment")
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
VIS_DIR = os.path.join(EXP_ROOT, "visualizations")
INVALID_VIS_DIR = os.path.join(EXP_ROOT, "invalid_visualizations")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e1_pose_results.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "summary.txt")

# ===== SETTINGS =====
ANGLE_THRESHOLD = 10.0
VIS_THRESHOLD = 0.5
TEXT_HEIGHT_FRAC = 1.0 / 50.0   # text height = 1/50 image height

# ===== MEDIAPIPE JOINTS =====
mp_pose = mp.solutions.pose
POSE_KNEE = {"left": mp_pose.PoseLandmark.LEFT_KNEE, "right": mp_pose.PoseLandmark.RIGHT_KNEE}
POSE_ANKLE = {"left": mp_pose.PoseLandmark.LEFT_ANKLE, "right": mp_pose.PoseLandmark.RIGHT_ANKLE}
POSE_HEEL = {"left": mp_pose.PoseLandmark.LEFT_HEEL, "right": mp_pose.PoseLandmark.RIGHT_HEEL}

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def ensure_and_clean_dirs():
    """Create experiment folders and clear contents before saving results."""
    for d in [RESULTS_DIR, VIS_DIR, INVALID_VIS_DIR]:
        os.makedirs(d, exist_ok=True)
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception:
                pass

def load_labels():
    """Load manual angles and labels from LABELS_PATH."""
    df = pd.read_csv(LABELS_PATH)
    labels = {}
    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        labels[fname] = {"manual_angle": float(row["angle_deg"]), "manual_label": str(row["label"]).strip()}
    return labels

def load_coords():
    """Load stance side information from COORDS_PATH (JSON)."""
    with open(COORDS_PATH, "r") as f:
        return json.load(f)

def px_from_norm(lmk, w, h):
    """Convert normalized landmark (0-1) to coordinates and visibility."""
    return (lmk.x * w, lmk.y * h, lmk.visibility)

def get_stance_landmarks(results, side, w, h):
    """Return (knee, ankle, heel) coordinates in the photo for the given stance side."""
    if results.pose_landmarks is None:
        return None, None, None
    lms = results.pose_landmarks.landmark
    try:
        knee = px_from_norm(lms[POSE_KNEE[side]], w, h)
        ankle = px_from_norm(lms[POSE_ANKLE[side]], w, h)
        heel = px_from_norm(lms[POSE_HEEL[side]], w, h)
    except Exception:
        return None, None, None
    return knee, ankle, heel

def compute_signed_angle_deg(knee, ankle, heel):
    """Compute signed ankle angle (in degrees) between knee→ankle and ankle→heel."""
    kx, ky, _ = knee; ax, ay, _ = ankle; hx, hy, _ = heel
    v1 = np.array([kx - ax, ky - ay], dtype=np.float32)
    v2 = np.array([ax - hx, ay - hy], dtype=np.float32)
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return None
    cross_z = (v1[0]*v2[1] - v1[1]*v2[0])
    dot = float(np.dot(v1, v2))
    angle = math.degrees(math.atan2(abs(cross_z), dot))
    sign = 1.0 if cross_z >= 0 else -1.0
    return sign * angle

def classify_label_from_angle(angle_deg):
    """Return 'overpronation' only if angle > ANGLE_THRESHOLD, else 'normal'."""
    return "overpronation" if angle_deg > ANGLE_THRESHOLD else "normal"

def draw_top_banner(img, lines, text_color=(255,255,255), error=False):
    """Draw a text banner at the top of the image."""
    h, w = img.shape[:2]
    text_px = max(12, int(h * TEXT_HEIGHT_FRAC))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, text_px / 30.0)
    thick = max(1, int(round(scale * 2/3)))
    line_heights = []
    for line in lines:
        (_, th), _ = cv2.getTextSize(line, font, scale, thick)
        line_heights.append(th + 4)
    banner_h = int(sum(line_heights) + 10)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    y = 6 + line_heights[0]
    for i, line in enumerate(lines):
        color = (0, 0, 255) if (error and i == 0) else text_color
        cv2.putText(img, line, (8, y), font, scale, color, thick, cv2.LINE_AA)
        y += line_heights[i]

def draw_keypoints_and_vectors(img, knee, ankle, heel):
    """Draw the stance leg's knee/ankle/heel points and connecting vectors."""
    def color_by_vis(v): return (0,255,0) if v >= VIS_THRESHOLD else (0,0,255)
    h, w = img.shape[:2]
    r = max(3, int(round(min(w, h) * 0.005)))
    kx, ky, kv = knee; ax, ay, av = ankle; hx, hy, hv = heel
    cv2.line(img, (int(kx), int(ky)), (int(ax), int(ay)), (255,255,255), max(1,r//2), cv2.LINE_AA)
    cv2.line(img, (int(ax), int(ay)), (int(hx), int(hy)), (255,255,255), max(1,r//2), cv2.LINE_AA)
    cv2.circle(img, (int(kx), int(ky)), r, color_by_vis(kv), -1, cv2.LINE_AA)
    cv2.circle(img, (int(ax), int(ay)), r, color_by_vis(av), -1, cv2.LINE_AA)
    cv2.circle(img, (int(hx), int(hy)), r, color_by_vis(hv), -1, cv2.LINE_AA)

def write_results_csv(rows):
    """Write all comparison results to RESULTS_CSV."""
    cols = ["filename","side","manual_angle","mediapipe_angle","error_deg","manual_label","mediapipe_label"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(by="filename", key=lambda s: s.map(lambda n: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', n)]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

def summarize_and_print(rows):
    """Compute mean absolute error, correlation, accuracy; print and save to summary.txt."""
    valid = [r for r in rows if isinstance(r[3], (int,float,np.floating))]
    total, n_valid = len(rows), len(valid)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines_out = [
        f"Experiment Timestamp: {ts}",
        f"ANGLE_THRESHOLD={ANGLE_THRESHOLD}°, VIS_THRESHOLD={VIS_THRESHOLD}",
        f"Processed: {total} total, {n_valid} valid"
    ]
    if n_valid == 0:
        lines_out.append("No valid detections to summarize.")
        summary = "\n".join(lines_out)+"\n"
        with open(SUMMARY_TXT,"w") as f:f.write(summary)
        print(summary.strip()); return
    manual = np.array([r[2] for r in valid], dtype=float)
    mp_ang = np.array([r[3] for r in valid], dtype=float)
    mae = float(np.mean(np.abs(mp_ang - manual)))
    corr = float("nan") if np.std(manual)<1e-9 or np.std(mp_ang)<1e-9 else float(np.corrcoef(manual, mp_ang)[0,1])
    agree = sum(1 for rrow in valid if str(rrow[5]).lower()==str(rrow[6]).lower() and str(rrow[6]).lower() in ("normal","overpronation"))
    acc = 100.0 * agree / n_valid if n_valid>0 else float("nan")
    lines_out += [f"Mean absolute error: {mae:.2f} deg", f"Pearson correlation (r): {corr:.3f}", f"Classification agreement: {acc:.1f}%"]
    summary = "\n".join(lines_out)+"\n"
    with open(SUMMARY_TXT,"w") as f:f.write(summary)
    print(summary.strip())

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    """Run the E1 experiment: process all photos, compute angles, and compare results."""
    ensure_and_clean_dirs()
    if not os.path.isfile(LABELS_PATH): raise FileNotFoundError(f"{LABELS_PATH} not found.")
    if not os.path.isfile(COORDS_PATH): raise FileNotFoundError(f"{COORDS_PATH} not found.")
    labels_map, coords_map = load_labels(), load_coords()
    raw_files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))], key=lambda n: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', n)])
    if not raw_files:
        print(f"No images found in {RAW_DIR}")
        return

    rows = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for fname in raw_files:
            img_path = os.path.join(RAW_DIR, fname)
            if not os.path.isfile(img_path): continue
            if fname not in labels_map or fname not in coords_map:
                img = cv2.imread(img_path)
                if img is not None:
                    draw_top_banner(img,[f"ERROR: Missing metadata for {fname}"],error=True)
                    cv2.imwrite(os.path.join(INVALID_VIS_DIR,f"{os.path.splitext(fname)[0]}_cmp.jpg"),img)
                manual_angle = labels_map.get(fname,{}).get("manual_angle","")
                manual_label = labels_map.get(fname,{}).get("manual_label","")
                side = coords_map.get(fname,{}).get("side","")
                rows.append([fname,side,manual_angle,"invalid","",manual_label,"invalid"])
                continue

            manual_angle = labels_map[fname]["manual_angle"]
            manual_label = labels_map[fname]["manual_label"]
            side = coords_map[fname].get("side","").lower()
            if side not in ("left","right"): side=""
            img = cv2.imread(img_path)
            if img is None:
                rows.append([fname,side,manual_angle,"invalid","",manual_label,"invalid"])
                continue

            h,w = img.shape[:2]
            rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            knee,ankle,heel = get_stance_landmarks(results,side,w,h)
            invalid,error_line=False,""
            if knee is None or ankle is None or heel is None:
                invalid,error_line=True,f"ERROR: Missing keypoints for {fname}"
            else:
                if (knee[2]<VIS_THRESHOLD) or (ankle[2]<VIS_THRESHOLD) or (heel[2]<VIS_THRESHOLD):
                    invalid,error_line=True,f"ERROR: Low visibility (<{VIS_THRESHOLD}) for {fname}"

            vis_img = img.copy()
            # Always draw joints (red if visibility < threshold)
            if knee is not None and ankle is not None and heel is not None:
                draw_keypoints_and_vectors(vis_img, knee, ankle, heel)

            # Compute MediaPipe angle only if valid
            if not invalid:
                mp_angle = compute_signed_angle_deg(knee, ankle, heel)

                # Flip sign for right stance leg so that overpronation ankle degrees are positive
                if side == "right" and mp_angle is not None:
                    mp_angle *= -1

                if mp_angle is None:
                    invalid,error_line=True,f"ERROR: Degenerate vectors for {fname}"

            if invalid:
                if knee is not None and ankle is not None and heel is not None:
                    k_v, a_v, h_v = knee[2], ankle[2], heel[2]
                    lines = [error_line, f"Visibilities: knee={k_v:.2f}, ankle={a_v:.2f}, heel={h_v:.2f}"]
                else:
                    lines = [error_line, "Visibilities: unavailable"]
                draw_top_banner(vis_img, lines, error=True)
                cv2.imwrite(os.path.join(INVALID_VIS_DIR, f"{os.path.splitext(fname)[0]}_cmp.jpg"), vis_img)
                rows.append([fname, side, manual_angle, "invalid", "", manual_label, "invalid"])
                continue

            mp_label = classify_label_from_angle(mp_angle)
            err = abs(mp_angle - manual_angle)
            k_v,a_v,h_v = knee[2], ankle[2], heel[2]
            lines = [
                f"Manual: {manual_angle:.2f} deg ({manual_label})",
                f"MediaPipe: {mp_angle:.2f} deg ({mp_label})",
                f"Visibilities: knee={k_v:.2f}, ankle={a_v:.2f}, heel={h_v:.2f}"
            ]
            draw_top_banner(vis_img, lines)
            cv2.imwrite(os.path.join(VIS_DIR,f"{os.path.splitext(fname)[0]}_cmp.jpg"),vis_img)
            rows.append([fname, side, manual_angle, float(f"{mp_angle:.6f}"), float(f"{err:.6f}"), manual_label, mp_label])

    write_results_csv(rows)
    summarize_and_print(rows)

if __name__ == "__main__":
    main()
