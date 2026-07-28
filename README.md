# GhibliFilter

## Project Overview

This project implements a CycleGAN to translate images between Ghibli-style and real-world photos using PyTorch. It includes custom dataset handling, generator and discriminator models, and training utilities.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Ghilbi
```

### 2. Install Dependencies

Recommended Python packages:

```bash
pip install torch torchvision albumentations numpy pillow tqdm matplotlib scikit-image opencv-python
```

### 3. Prepare Data

- Place your training images in `data/train/ghilbi/` and `data/train/real/`.
- Validation images go in `data/val/ghilbi/` and `data/val/real/`.
- The data folder is ignored by git (see `.gitignore`).

### 4. Training

Run the training script:

```bash
python train.py
```

Checkpoints will be saved as `.pth.tar` files and images in `saved_images/`.

### 5. Checkpoints & Images

- Checkpoints are saved every epoch and contain model and optimizer states.
- Generated images are saved every 100 batches in `saved_images/`.

### 6. Resume Training

To resume from a checkpoint, set `LOAD_MODEL = True` in `config.py`.

### 7. Customization

- Adjust hyperparameters in `config.py`.
- Edit transforms in `config.py` for different image sizes or augmentations.

## File Structure

- `config.py`: Configuration and transforms
- `dataset.py`: Custom dataset loader
- `generator_model.py`, `discriminator_model.py`: Model definitions
- `train.py`: Training loop
- `utils.py`: Checkpoint utilities

## Notes

- Make sure your images are RGB and of similar size.
- Training GANs may require many epochs for good results.

## Architecture
[block_diag.pdf](https://github.com/user-attachments/files/30478930/block_diag.pdf)

### Sample output
<img width="1606" height="852" alt="Screenshot from 2025-09-23 08-10-56" src="https://github.com/user-attachments/assets/f77e609a-8231-4a28-bfa1-e398113a41e2" />


