# Antarctic Segmentation

This repository contains code for **semantic segmentation of historical cryospheric aerial imagery** using U-Net models. It was developed as part of a PhD project at TU Delft to classify surface types in historical photographs of the Antarctic Peninsula.

The code accompanies two publications:

- **Revisiting the Past: A comparative study for semantic segmentation of historical images of Adelaide Island using U-nets** — *ISPRS Open Journal of Photogrammetry and Remote Sensing* ([download](https://www.sciencedirect.com/science/article/pii/S2667393223000273))
- **Semantic segmentation of historical photographs of the Antarctic Peninsula** — *ISPRS 2022, Nice* ([download](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2022/237/2022/))

![Example for segmentation](https://github.com/fdahle/antarctic_segmentation/blob/main/readme/segmentation_example.png?raw=true)

---

## Datasets

The source images are [publicly available](https://www.pgc.umn.edu/data/aerial/) from the USGS TMA archive and can be downloaded [here](https://data.pgc.umn.edu/aerial/usgs/tma/photos/).

Due to file size limitations, the training data and segmented imagery for Adelaide Island are not hosted on GitHub. They can be downloaded from:  
**https://doi.org/10.4121/de8ea9d4-f986-41fc-9412-6765985a0c9c**

---

## Installation

**Requirements:** Python 3.8+

Install the required packages with:

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-learn scikit-image rasterio albumentations pytorch-msssim skmultilearn scipy Pillow
```

> **Note:** For GPU support, install the appropriate CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Segmentation Classes

The model predicts the following 7 semantic classes:

| ID | Class   |
|----|---------|
| 1  | Ice     |
| 2  | Snow    |
| 3  | Rocks   |
| 4  | Water   |
| 5  | Clouds  |
| 6  | Sky     |
| 7  | Unknown |

---

## Project Structure

```
antarctic_segmentation/
├── train_model.py          # Train a U-Net segmentation model
├── apply_model.py          # Apply a trained model to new images
├── evaluate_model.py       # Visualise training statistics from a saved model
├── segmentator.py          # GUI tool for manually refining segmented images
├── base_functions/         # Core utility functions (image loading, edge removal, etc.)
├── classes/                # U-Net model definitions and loss functions
│   ├── u_net.py            # Full U-Net (4 encoder/decoder levels)
│   ├── u_net_small.py      # Compact U-Net (3 encoder/decoder levels)
│   ├── image_data_set.py   # PyTorch Dataset with augmentation pipeline
│   ├── dice_loss.py        # Dice loss implementation
│   └── focal_loss.py       # Focal loss implementation
└── sub_functions/          # Helper functions (unsupervised segmentation, etc.)
```

---

## Usage

### Training

Open `train_model.py` and set the paths and parameters at the top of the file:

```python
path_folder_images    = "/path/to/raw/images"
path_folder_segmented = "/path/to/labelled/images"
path_folder_models    = "/path/to/save/models"

params_training = {
    "model_name":   "my_model",
    "model_type":   "normal",   # "normal" or "small"
    "max_epochs":   500,
    "learning_rate": 0.001,
    "loss_type":    "cross_entropy",  # "cross_entropy", "focal", "dice", or "ssim"
    "output_layers": 6,               # number of classes
    ...
}
```

Then run:

```bash
python train_model.py
```

Key command-line arguments are also available:

```bash
python train_model.py --model_name my_model --loss_type focal --batch_size 8
```

The script saves a `.pth` checkpoint and a `.json` statistics file after each epoch (configurable with `save_step`).

---

### Applying a Model

Open `apply_model.py`, set the paths and the run mode, then execute:

```bash
python apply_model.py
```

| Variable        | Description                                              |
|-----------------|----------------------------------------------------------|
| `select_type`   | `"random"` to pick random images, `"ids"` for specific ones |
| `num_images`    | Number of images if `select_type="random"`               |
| `image_ids`     | List of image IDs if `select_type="ids"`                 |
| `path_folder_images` | Path to the folder with raw images               |
| `path_folder_models` | Path to the folder containing the trained model  |
| `model_id`      | Model filename (without extension), or `"LATEST_PTH"`    |

---

### Evaluating a Model

`evaluate_model.py` loads the `.json` statistics file produced during training and plots the loss, accuracy, and F1-score curves.

```bash
python evaluate_model.py
```

Set `model_name` and `model_folder` at the top of the file before running.

---

### Segmentator GUI

`segmentator.py` provides a Tkinter-based GUI for manually refining segmented images (e.g. output from unsupervised segmentation). Set the `path_images` and `path_segmented` entries in `segmentator_params` at the top of the file and run:

```bash
python segmentator.py
```

---

## Background

This project is part of [Felix Dahle's](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/staff/phd-students/f-felix-felix-dahle) PhD research at TU Delft, aiming to reconstruct 3D models of Antarctica from historical aerial photography using Structure from Motion. More code for the broader pipeline can be found at the [Antarctic TMA repository](https://github.com/fdahle/Antarctic_TMA/).

---

## Citation

If you use this code, please cite it as described in [CITATION.cff](CITATION.cff), or use one of the papers listed above.
