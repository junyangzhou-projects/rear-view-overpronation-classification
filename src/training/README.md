# Training Scripts

This folder contains three sequential training and evaluation phases for the overpronation classification model:

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `phase1_train_headonly.py` | Train a frozen ResNet-18 (head-only) classifier using prepared and augmented stance-leg/foot crops. |
| 2 | `phase2_finetune_resnet18.py` | Fine-tune the Phase 1 model with grouped 5-fold cross-validation; unfreeze backbone layers after warm-up for better feature adaptation. |
| 3 | `phase3_eval_visualize.py` | Evaluate and summarize Phase 2 results; generate Grad-CAM visualizations to display model focus on stance ankle (desired) and other regions. |

Each script contains full documentation in its header comments.
