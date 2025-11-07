# Source Code

This folder contains all scripts for the overpronation classification project.

| Subfolder | Description |
|-----------|-------------|
| `preprocessing/` | Scripts for assigning labels, collecting coordinates, cropping stance-side leg/foot regions, and generating augmented images. |
| `experiments/` | Scripts for the pose-angle (MediaPipe) experiment and analysis. |
| `training/` | Scripts for the three sequential training and evaluation phases of the overpronation classifier. |

More detailed documentation is provided in the README files within each subfolder.  
The scripts are designed to be run in the **preprocessing → experiments → training** sequence, and the **numerical order** within each subfolder should be followed when executing the scripts.
