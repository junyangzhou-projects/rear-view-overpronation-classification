# Rear-View Overpronation Classification

This project develops a computer vision pipeline for detecting **overpronation** from rear-view running photos. It uses a fine-tuned ResNet-18 classifier to identify whether a runner’s stance-leg (with the foot mostly or fully touching the ground) shows overpronation or normal alignment of the ankle joint. The project emphasizes interpretability and reproducibility through a fully documented, multi-phase workflow.

---

### 1. Project Overview

Overpronation occurs when the ankle rolls inward excessively during running or other physical activities, which can increase the risk of injury. For runners and people with an active lifestyle, identifying whether they overpronate is important for adjusting running form, training plans, and choosing the right type of footwear or equipment. However, **it is sometimes difficult to determine overpronation by visual inspection**, especially for people who are not familiar with the term.

This project was motivated by the question:  
> *Is it possible to detect overpronation automatically using computer vision techniques applied to rear-view running photos?*

To simplify the scope, this project adopts a **geometric proxy definition** rather than a medical one:  
> Overpronation is assumed to occur when, on the stance-side, the angle between knee→ankle and ankle→heel exceeds **+10 degrees**, indicating that the foot is rotated inward.

This definition is used only for experimental purposes to test whether visual features corresponding to this geometry can be detected by a neural network.  
**It is not a medical diagnostic criterion** — anyone suspecting overpronation or other gait related issues should consult a qualified medical or physiotherapy professional.

---

### 2. Dataset

The dataset consists of **65 manually collected rear-view running photos**, each showing a runner with a single stance-side foot on the ground. Images were selected to ensure clear visibility of the entire body.

Ground-truth labels were derived from **measured stance-side ankle angles**. A third-party measurement tool were used to determine the knee–ankle–heel angle, and these values were entered using the preprocessing script to generate the metadata file `labels.csv` for training and validation.

For public release, two example photos (with no faces) from Pixabay are included in `data/raw/`. The folder also provides corresponding metadata, prepared crops, and augmented images showing what kind of data were used for model training.

See [`data/README.md`](data/README.md) for the complete folder structure and photo attribution.

---

### 3. Pipeline Summary

This project follows a three-stage pipeline that processes rear-view running photos from raw inputs to interpretable model outputs.

| Stage | Folder | Description |
|-------|--------|-------------|
| **Preprocessing** | `src/preprocessing/` | Prepare the dataset by assigning manual labels, collecting head and lateral heel coordinates, and cropping stance-side leg/foot regions. This stage outputs prepared and augmented images, which serve as the model inputs. |
| **Experiments** | `src/experiments/` | Validate a fully pose-based pipeline by comparing MediaPipe-estimated ankle angles/labels with manually measured ones and visualizing joint keypoints. |
| **Training** | `src/training/` | Train and evaluate ResNet-18 models in three sequential phases: (1) head-only training, (2) fine-tuning with 5-fold cross-validation, and (3) evaluation with Grad-CAM visualizations. |

Scripts are designed to be executed in the order **preprocessing → experiments → training**. Within each folder, follow the numerical sequence described in its README.

---

### 4. Experiment Motivation

Three potential methods for classifying overpronation were evaluated conceptually:

1. **Pose-only classification:**  
   Use MediaPipe pose detection to estimate ankle angles and directly assign labels. *However*, as shown in `data/experiments/overpronation_sample_cmp`, the detected ankle → heel line can often be unreliable — in some cases, the entire stance-leg is detected outside the body boundary even when the photo is clear. Therefore, this method is proved to be unreliable.

2. **Pose-guided cropping + CNN:**  
   Use MediaPipe solely to crop the stance-side leg/foot region, then train a CNN on the cropped region. The instability of pose detection can again limit model performance.

3. **Manual cropping + CNN (adopted approach):**  
   A custom preprocessing tool was developed to manually crop the stance-side leg/foot region for each photo. These manually prepared crops form the basis of the CNN classifier training. Because only 65 raw photos are collected, the lightweight and pretrained ResNet18 is selected.

---

### 5. Key Results and Model Evaluation

| Phase | Validation AUC | Validation Accuracy |
|:------|:----------------|:--------------------|
| **Phase 1 (Head-Only)** | ≈ 0.73 | ≈ 0.62 |
| **Phase 2 (Fine-Tuned)** | **0.79 ± 0.14** | **0.63 ± 0.18** |

Fine-tuning the ResNet-18 backbone improved the **validation AUC** from ≈ 0.73 to **0.79 ± 0.14**, indicating stronger classification performance between the overpronation and normal classes.

**Grad-CAM visualizations** show that correctly classified samples often exhibit high activation near the **stance-side ankle regions**, matching expected biomechanical cues. Misclassified or uncertain cases often show dispersed or out-of-body activations, suggesting that the model may reach for background or irrelevant regions when ankle features are ambiguous. These findings highlight both the improvement achieved through fine-tuning and potential next steps — such as exploring refined classification methods or threshold calibration — to further improve validation accuracy.

---

### 6. Repository Structure

```
data/                → Example dataset and output structures
src/preprocessing/   → Data labeling, coordinate collection, cropping, augmentation
src/experiments/     → Pose-angle pipeline validation and visualization
src/training/        → Lightweight CNN training (Phases 1–3)
```

Each folder above includes a detailed README explaining its scripts and usage.

---

### 7. How to Run (in order)

1. **Install dependencies**  
   Install the required libraries.

2. **Run preprocessing scripts**  
   Execute in order (`pre1_`, `pre2_`, `pre3_`) to prepare cropped and augmented data.

3. **(Optional)** Run the pose-angle experiment in `src/experiments/` to check MediaPipe angles.

4. **Run training phases** sequentially:  
   - Phase 1: `phase1_train_headonly.py`  
   - Phase 2: `phase2_finetune_resnet18.py`  
   - Phase 3: `phase3_eval_visualize.py`

5. **View outputs**  
   Results are saved under `data/training/phase*/` subfolders.

---

### 8. Environment

- **Python:** 3.9.6  
- **Core libraries:** PyTorch 2.8, torchvision 0.23, MediaPipe 0.10.21, OpenCV 4.11.0, scikit-learn 1.6, tqdm 4.67

---

### 9. License & Credits

MIT licensed.

Example images © Pixabay photographers:  
- [jotoya](https://pixabay.com/users/jotoya-1781229/) — *Running on the road*  
- [TheOtherKev](https://pixabay.com/users/theotherkev-1921669/) — *Running in the park*  

Both are licensed under the **Pixabay License** (free for commercial and non-commercial use, no attribution required, and modification allowed).

---

### 10. Summary

This repository demonstrates a complete **small-data computer-vision workflow** for assessing rear-view running gait and detecting overpronation. Future extensions may include expanding the dataset, improving automatic stance-side leg/foot localization (e.g., testing models beyond MediaPipe), refining classification or thresholding methods, and performing quantitative comparisons between pose-based and CNN-based approaches.
