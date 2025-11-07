"""
Preprocessing Step 1 — Assign & Verify Photo Labels
---------------------------------------------------
- Check if data/metadata/labels.csv exist.

- For each runner's photo in data/raw/:
    - If a entry has no corresponding photo -> remove the entry.
    - If a photo has no entry in labels.csv -> let user to enter the manually 
      measured angle between [lower leg to ankle] and [ankle to heel] -> assign
      label "overpronation" if angle > ANGLE_THRESHOLD (default at 10.0), else 
      assign "normal".

INPUT: data/raw, (data/metadata/labels.csv)
OUTPUT: data/metadata/labels.csv
"""

import os
import re
import csv

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
META_DIR = os.path.join(ROOT_DIR, "data/metadata")
LABELS_PATH = os.path.join(META_DIR, "labels.csv")

# ===== SETTINGS =====
ANGLE_THRESHOLD = 10.0

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def ensure_structure():
    """Make sure META_DIR exists and LABELS_PATH is initialized."""
    os.makedirs(META_DIR, exist_ok=True)
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"{RAW_DIR} does not exist.")
    if not os.path.isfile(LABELS_PATH):
        with open(LABELS_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["filename", "angle_deg", "label"])
        print(f"Created {LABELS_PATH}")

def load_labels():
    """Load existing labels from LABELS_PATH."""
    labels = {}
    if os.path.isfile(LABELS_PATH):
        with open(LABELS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fname = row["filename"].strip().lstrip("\ufeff")
                    labels[fname] = {
                        "angle_deg": float(row["angle_deg"]),
                        "label": row["label"].strip()
                    }
                except (KeyError, ValueError):
                    continue
    return labels

def save_labels(labels):
    """Write labels dictionary back to LABELS_PATH (sorted numerically)."""
    with open(LABELS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_deg", "label"])
        for fname in sorted(labels.keys(), key=natural_key):
            info = labels[fname]
            writer.writerow([fname, info["angle_deg"], info["label"]])
    print(f"Updated labels saved -> {LABELS_PATH}")

def natural_key(name):
    """Return a list for natural sorting of filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    ensure_structure()
    labels = load_labels()

    raw_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], key=natural_key)

    if not raw_files:
        print(f"No image files found in {RAW_DIR}")
        return

    # Remove entries having no corresponding photo
    existing = set(raw_files)
    removed = [f for f in list(labels.keys()) if f not in existing]
    for f in removed:
        del labels[f]
    if removed:
        print(f"Removed stale entries: {removed}")

    # Add entries for new photos
    for fname in raw_files:
        if fname not in labels:
            while True:
                try:
                    val = input(f"Enter ankle angle (°) for {fname}: ").strip()
                    angle = float(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")
            label = "overpronation" if angle > ANGLE_THRESHOLD else "normal"
            labels[fname] = {"angle_deg": angle, "label": label}
            print(f"-> {fname}: {angle:.1f}° -> {label}")

    save_labels(labels)
    print("Label verification complete.")

if __name__ == "__main__":
    main()
