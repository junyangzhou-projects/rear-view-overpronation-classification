# Data Folder

This folder contains example data and output (folder) structures for all **preprocessing**, **experiments**, and **training** stages of the overpronation classification project.

| Folder | Description |
|--------|-------------|
| `raw/` | Two example rear-view running photos (from Pixabay, no faces) used to demonstrate preprocessing. |
| `metadata/` | Example label (`labels.csv`) and coordinate (`coords.json`) files generated from preprocessing. |
| `prepared/` | Example cropped and padded stance-leg/foot regions generated from the raw photos; represents part of the model input. |
| `augmented/` | Example augmented and padded images (rotation, brightness, blur, etc.); also represents part of the model input. |
| `experiments/e1_pose_angle_experiment/` | Example output for the pose-angle (MediaPipe) experiment, including a sample CSV and visualization. |
| `training/phase1_train_headonly/` | Placeholder for Phase 1 training outputs (head-only ResNet-18). |
| `training/phase2_finetune_resnet18/` | Placeholder for Phase 2 fine-tuning outputs (5-fold cross-validation). |
| `training/phase3_eval_visualize/` | Placeholder for Phase 3 evaluation and Grad-CAM visualization results. |

---

### Example Raw Photos

| Label | File | Source |
|-------|------|--------|
| Overpronation | `overpronation_sample.jpg` | [Pixabay – Running on the road](https://pixabay.com/photos/man-running-exercise-jogging-road-5131486/) <br> *Photo by [jotoya](https://pixabay.com/users/jotoya-1781229/)* |
| Normal | `normal_sample.jpg` | [Pixabay – Running in the park](https://pixabay.com/photos/runner-jogging-run-in-the-park-4356279/) <br> *Photo by [TheOtherKev](https://pixabay.com/users/theotherkev-1921669/)* |

Both photos are licensed under the **Pixabay License** — free for commercial and
noncommercial use, no attribution required, and modification allowed.  
Faces are not visible in the example photos.

---

Each folder represents one stage of the **preprocessing → experiments → training**
pipeline.
