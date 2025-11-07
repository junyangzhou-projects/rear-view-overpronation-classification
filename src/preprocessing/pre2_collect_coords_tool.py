"""
Preprocessing Step 2 — Collect Head & Heel Coordinates and stance side
---------------------------------------------------
- Check if data/metadata/coords.json exist.

- For each runner's photo in data/raw/:
    - If a entry has no corresponding photo -> remove the entry.
    - If a photo has no entry in coords.json -> display the photo and let user:
        1) Left-click the top of head (in green mark).
        2) Left-click the lateral heel edge (in red mark).
        3) Press "l" or "r" to set the stance side (auto save and proceed).
        4) Press "d" to delete input for the current photo and redo.
        5) Press esc or "q" to exit the annotation tool.

INPUT: data/raw
OUTPUT: data/metadata/coords.json
"""

import os
import re
import json
import cv2

# ===== PATHS =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
META_DIR = os.path.join(ROOT_DIR, "data/metadata")
COORDS_PATH = os.path.join(META_DIR, "coords.json")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def ensure_structure():
    """Make sure META_DIR exists and COORDS_PATH is initialized."""
    os.makedirs(META_DIR, exist_ok=True)
    if not os.path.isfile(COORDS_PATH):
        with open(COORDS_PATH, "w") as f:
            json.dump({}, f, indent=2)
        print(f"Created {COORDS_PATH}")

def load_coords():
    """Load existing coordinates from COORDS_PATH."""
    with open(COORDS_PATH) as f:
        return json.load(f)

def save_coords(coords):
    """Write coordinates dictionary back to COORDS_PATH."""
    with open(COORDS_PATH, "w") as f:
        json.dump(coords, f, indent=2)
    print(f"Updated coordinates saved -> {COORDS_PATH}")

def natural_key(name):
    """Return a list for natural sorting of filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    ensure_structure()
    coords = load_coords()

    raw_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], key=natural_key)

    if not raw_files:
        print(f"No image files found in {RAW_DIR}")
        return

    # Remove entries having no corresponding photo
    existing = set(raw_files)
    removed = [f for f in list(coords.keys()) if f not in existing]
    for f in removed:
        del coords[f]
    if removed:
        print(f"Removed stale entries: {removed}")
        save_coords(coords)

    # Add entries for new photos
    for fname in raw_files:
        if fname in coords:
            continue  # skip already annotated

        img_path = os.path.join(RAW_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot load {fname}")
            continue

        display = img.copy()
        click_points = []
        side = None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_points.append((x, y))
                color = (0, 255, 0) if len(click_points) == 1 else (0, 0, 255)
                cv2.circle(display, (x, y), 6, color, -1)
                cv2.imshow(window_name, display)

        window_name = "Annotate (click TOP OF HEAD adn LATERAL HEEL EDGE; then input l/r for stance side)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)

        print(f"Annotating {fname} ...")
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            # Delete and redo
            if key == ord("d"):
                click_points.clear()
                display = img.copy()
                side = None
                cv2.imshow(window_name, display)
                print("Redo current image.")

            # Set stance side and auto save
            elif key in [ord("l"), ord("r")]:
                side = "left" if key == ord("l") else "right"
                print(f"Side: {side}")

                if len(click_points) == 2:
                    coords[fname] = {
                        "top": [int(click_points[0][0]), int(click_points[0][1])],
                        "bottom": [int(click_points[1][0]), int(click_points[1][1])],
                        "side": side
                    }
                    save_coords(coords)
                    print(f"-> Saved {fname}: side={side}, top={click_points[0]}, bottom={click_points[1]}")
                    break
                else:
                    print("Need 2 points (head and heel) before saving.")

            # Quit
            elif key in [27, ord("q")]:  # Esc or q
                print("Exiting annotation tool...")
                save_coords(coords)
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    print("All coordinates collected.")

if __name__ == "__main__":
    main()
