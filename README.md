# Apple Leaf Disease Classification with CNNs

Can a small convolutional neural network reliably tell apart three apple leaf diseases and healthy leaves from a photo? This project trains and compares two CNN architectures in PyTorch on the apple subset of the PlantVillage dataset.

## Highlights

- **Dataset**: 3,171 RGB images (256x256) in 4 classes, resized to 128x128 and split 70/15/15 (2,219 / 475 / 477) with a fixed seed
- **Class imbalance**: healthy 1,645 vs. Cedar apple rust 275 images (6.29x imbalance in the training split), so models are evaluated with **macro precision and macro recall**
- **Two architectures**: a shallow CNN with a large fully connected head (16.8M parameters) and a deeper CNN with BatchNorm and global average pooling (94K parameters)
- **16 experiments**: 2 architectures x 2 learning rates (0.001, 0.0005) x 4 schedulers (none, StepLR, ReduceLROnPlateau, CosineAnnealingLR), 30 epochs each, Adam + cross-entropy
- **Best model**: CNN_Model2, LR 0.001, ReduceLROnPlateau, with **0.9895 macro precision / 0.9868 macro recall** on the test set, about 180x fewer parameters than Model 1

## Results

Test set, best configuration per architecture, compared with two heuristic baselines (baselines evaluated on the validation set):

| Model | Configuration | Macro Precision | Macro Recall |
|---|---|---|---|
| CNN_Model2 (BatchNorm + GAP, 94,212 params) | LR 0.001, ReduceLROnPlateau | **0.9895** | **0.9868** |
| CNN_Model1 (2 conv blocks + FC, 16,797,892 params) | LR 0.0005, CosineAnnealingLR | 0.9606 | 0.9538 |
| Baseline: stratified random | – | 0.2366 | 0.2465 |
| Baseline: most frequent class | always "healthy" | 0.1226 | 0.2500 |

All 16 CNN runs reached at least 0.944 macro precision on the test set. All 8 CNN_Model2 runs scored higher than all 8 CNN_Model1 runs. Model 1 overfits clearly: its training loss falls close to zero while validation loss rises.

## Dataset

The images are **not included** in this repository. The project uses the **PlantVillage** dataset (apple subset, 4 classes):

- Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (use the `color` version)
- Also available via [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

Download the dataset and copy the four apple folders into a `data/` folder next to the notebook:

```
apple-leaf-disease-classification/
├── report.ipynb
└── data/
    ├── Apple___Apple_scab/          # 630 images
    ├── Apple___Black_rot/           # 621 images
    ├── Apple___Cedar_apple_rust/    # 275 images
    └── Apple___healthy/             # 1,645 images
```

The notebook reads the images from `data/` (`image_root = "data"`) with `torchvision.datasets.ImageFolder`, which uses each subfolder as a class. Keeping the images in their own folder stops other folders (such as `.git`) from being picked up as classes. `data/` is listed in `.gitignore`.

## Trained models

Both best checkpoints (PyTorch `state_dict`) are included:

| File | Model | Note |
|---|---|---|
| `cnn_model1_best.zip` | CNN_Model1, LR 0.0005, CosineAnnealingLR | **Zipped** because of its size (about 67 MB unzipped). Unzip it before loading. |
| `cnn_model2_best.pth` | CNN_Model2, LR 0.001, ReduceLROnPlateau | Ready to load |

```bash
unzip cnn_model1_best.zip        # creates cnn_model1_best.pth
```

```python
import torch
# CNN_Model1 and CNN_Model2 are defined in report.ipynb (Task 6)
model2 = CNN_Model2()
model2.load_state_dict(torch.load("cnn_model2_best.pth", map_location="cpu"))
model2.eval()

model1 = CNN_Model1()
model1.load_state_dict(torch.load("cnn_model1_best.pth", map_location="cpu"))  # after unzipping
model1.eval()
```

Inputs must be preprocessed as in the notebook: resize to 128x128, `ToTensor()`, normalize with ImageNet mean/std. Class order is `Apple_scab, Black_rot, Cedar_apple_rust, healthy`.

## Project structure

| File | Description |
|---|---|
| `report.ipynb` | Full project: data loading, IDA/EDA, baselines, both CNN architectures, 16 experiments, test evaluation, discussion |
| `cnn_model1_best.zip` | Best CNN_Model1 checkpoint (zipped) |
| `cnn_model2_best.pth` | Best CNN_Model2 checkpoint |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
# download the dataset into data/ as described above
jupyter notebook report.ipynb
```

A CUDA GPU is recommended. The notebook was run on CUDA and falls back to CPU, which works but makes the 16 training runs slow.

## Limitations and next steps

- PlantVillage images are taken under controlled lab conditions. Field photos (variable light, background, blur) will likely give lower accuracy.
- Cedar apple rust has only 182 training images and no class weighting was used.
- A single fixed split gives no estimate of variance. Repeating with other seeds or using cross-validation would show how robust the 0.9895 result is.
- The model has no out-of-distribution detection and will assign any image to one of the 4 classes.
- Next steps: targeted augmentation for the rare class, distillation to an even smaller model for edge devices, semi-supervised fine-tuning on unlabelled field images.

**Author:** Mehdi Gurbanzade · M.Sc. Data Science
