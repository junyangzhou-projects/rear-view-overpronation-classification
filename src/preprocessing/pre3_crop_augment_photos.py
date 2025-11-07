"""
Preprocessing Step 3 — Crop and Augment Photos
---------------------------------------------------
- If data/prepared/ or data/augmented/ already exist, clear their contents.
  Otherwise, create them. Then create the <label> subfolders.

- Use the preprocessed info from data/metadata/coords.json and 
  data/metadata/labels.csv to crop the stance leg/feet region from each photo 
  in data/raw/, then save:
    - Cropped and padded originals to data/prepared/<label>/.
    - Augmented (random choices from rotation, scale, brightness, contrast, 
      and blur) and padded versions to data/augmented/<label>/.

INPUT: data/raw, data/metadata/coords.json, data/metadata/labels.csv
OUTPUT: data/prepared/<label>/, data/augmented/<label>/
"""

import os
import re
import math
import json
import random
import shutil
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
META_DIR = os.path.join(ROOT_DIR, "data/metadata")
COORDS_PATH = os.path.join(META_DIR, "coords.json")
LABELS_PATH = os.path.join(META_DIR, "labels.csv")
PREPARED_ROOT = os.path.join(ROOT_DIR, "data/prepared")
AUGMENTED_ROOT = os.path.join(ROOT_DIR, "data/augmented")

# ===== SETTINGS =====
OUTPUT_SIZE = 224
AUGMENTATIONS_PER_IMAGE = 3
WIDTH_RATIO = 0.15
REPRODUCIBLE = True
RANDOM_SEED = 42

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def ensure_output_folders(classes):
    """Ensure prepared and augmented folders are empty and structured by class."""
    for root in [PREPARED_ROOT, AUGMENTED_ROOT]:
        if os.path.exists(root):
            shutil.rmtree(root)
            print(f"Cleared existing folder: {root}")
        os.makedirs(root, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(root, cls), exist_ok=True)
    print("Prepared and augmented directories are ready.\n")

def crop_leg_feet(img, info):
    """Crop stance leg/feet region using head (top) and heel (bottom) coordinates."""
    w_img, h_img = img.size
    x_top, y_top = info["top"]
    x_bottom, y_bottom = info["bottom"]
    side = info.get("side", "right").lower()

    body_height = abs(y_bottom - y_top)
    full_width = WIDTH_RATIO * body_height
    half_width = full_width / 2

    x_center = int(x_bottom)
    crop_left = max(0, int(x_center - half_width))
    crop_right = min(w_img, int(x_center + half_width))
    y_min = max(0, int(y_bottom - 0.25 * body_height))
    y_max = min(h_img, int(y_bottom + 0.05 * body_height))

    cropped = img.crop((crop_left, y_min, crop_right, y_max))
    if side == "left":
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
    return cropped

def safe_local_augmentation(img):
    """Apply safe augmentations without cutoff."""
    choice = random.choice(["rot", "scale", "bright", "contrast", "blur"])
    w, h = img.size

    if choice == "rot":
        deg = random.uniform(-10, 10)
        theta = math.radians(abs(deg))
        w_new = int(abs(w * math.cos(theta)) + abs(h * math.sin(theta)))
        h_new = int(abs(w * math.sin(theta)) + abs(h * math.cos(theta)))
        safe_w = max(w_new, w)
        safe_h = max(h_new, h)

        canvas = Image.new("RGB", (safe_w * 2, safe_h * 2), (0, 0, 0))
        cx, cy = canvas.size[0] // 2, canvas.size[1] // 2
        canvas.paste(img, (cx - w // 2, cy - h // 2))
        rotated = canvas.rotate(deg, resample=Image.BILINEAR, expand=True)
        rx, ry = rotated.size
        crop = rotated.crop((rx // 2 - safe_w // 2,
                             ry // 2 - safe_h // 2,
                             rx // 2 + safe_w // 2,
                             ry // 2 + safe_h // 2))
        return crop, f"_rot{int(deg)}"

    elif choice == "scale":
        factor = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * factor), int(h * factor)
        scaled = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        offset = ((w - new_w)//2, (h - new_h)//2)
        canvas.paste(scaled, offset)
        return canvas, f"_sc{round(factor,2)}"

    elif choice == "bright":
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Brightness(img).enhance(factor), f"_b{round(factor,2)}"

    elif choice == "contrast":
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Contrast(img).enhance(factor), f"_c{round(factor,2)}"

    elif choice == "blur":
        sigma = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma)), f"_bl{round(sigma,1)}"

def pad_to_square(img, size=224):
    """Pad an photo to a square shape with black background."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    delta_w, delta_h = size - new_w, size - new_h
    padding = (delta_w // 2, delta_h // 2,
               delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=(0, 0, 0))

def natural_key(name):
    """Return a list for natural sorting of filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    if REPRODUCIBLE:
        random.seed(RANDOM_SEED)
        print(f"Reproducible mode ON (seed={RANDOM_SEED})")
    else:
        print("Random augmentation mode ON (new random outputs each run)")

    # Load metadata
    with open(COORDS_PATH, "r") as f:
        coords = json.load(f)
    labels = pd.read_csv(LABELS_PATH)
    label_map = {row.filename: row.label.strip().lower() for _, row in labels.iterrows()}
    classes = sorted(list(set(label_map.values())))

    # Prepare output directories
    ensure_output_folders(classes)

    # List all photos
    all_images = sorted(
        [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_key
    )

    # Process all photos and save to the output folders
    total = len(all_images)
    for idx, filename in enumerate(all_images, start=1):
        if filename not in coords or filename not in label_map:
            print(f"Skipped (missing coords or label): {filename}")
            continue

        label = label_map[filename]
        img_path = os.path.join(RAW_DIR, filename)
        img = Image.open(img_path).convert("RGB")

        # --- Crop ---
        cropped = crop_leg_feet(img, coords[filename])

        # --- Save the cropped, padded, and non-augmented photos ---
        base_name, ext = os.path.splitext(filename)
        padded = pad_to_square(cropped, OUTPUT_SIZE)
        padded.save(os.path.join(PREPARED_ROOT, label, f"{base_name}_padded{ext}"))

        # --- Generate safe augmentations and save ---
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img, suffix = safe_local_augmentation(cropped)
            aug_padded = pad_to_square(aug_img, OUTPUT_SIZE)
            new_name = f"{base_name}{suffix}_{i}{ext}"
            aug_padded.save(os.path.join(AUGMENTED_ROOT, label, new_name))

        print(f"[{idx}/{total}] -> {filename} ({label})")

    print("\nAll images cropped, safely augmented, and padded by class.")
    print(f"Prepared set -> {PREPARED_ROOT}")
    print(f"Augmented set -> {AUGMENTED_ROOT}")

if __name__ == "__main__":
    main()
