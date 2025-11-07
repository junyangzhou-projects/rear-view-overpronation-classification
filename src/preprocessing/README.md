# Preprocessing Scripts

This folder contains three sequential preprocessing steps for the data:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `pre1_assign_photo_label.py` | Assign and verify photo labels using manually measured ankle angles. |
| 2 | `pre2_collect_coords_tool.py` | Interactively collect head and heel coordinates and stance side. |
| 3 | `pre3_crop_augment_photos.py` | Crop the stance leg/feet region and generate augmented images. |

Each script contains full documentation in its header comments.
