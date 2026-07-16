# Sentinel Image Matching

## Project overview

This project implements an automatic image matching pipeline for Sentinel-2 Level-1C satellite imagery using the pretrained **LoFTR (Detector-Free Local Feature Matching)** model.

The pipeline automatically:

- loads Sentinel-2 `.SAFE` scenes;
- constructs RGB images from spectral bands;
- splits large satellite images into fixed-size tiles;
- filters invalid tiles containing large NoData regions;
- matches corresponding tiles using the pretrained LoFTR model;
- selects the best matching tile pair;
- visualizes feature correspondences.

The implementation is fully automatic and can be applied to any Sentinel-2 Level-1C scenes stored in the standard `.SAFE` format.

# Project goal

The goal of this project is to automatically identify corresponding regions between two Sentinel-2 satellite images acquired on different dates using a modern detector-free feature matching approach.

Instead of training a model from scratch, the project uses the pretrained **LoFTR** model, which is specifically designed for robust local feature matching.

# Pipeline

The complete workflow consists of the following stages:

1. Locate the Sentinel-2 `IMG_DATA` directory.
2. Read RGB spectral bands:
   - B04 — Red
   - B03 — Green
   - B02 — Blue
3. Construct a natural-color RGB image.
4. Split each image into **1024 × 1024** pixel tiles.
5. Remove tiles containing large NoData regions.
6. Match corresponding tiles using the pretrained LoFTR model.
7. Rank candidate tile pairs according to:
   - number of feature matches;
   - average matching confidence.
8. Select the best matching tile pair.
9. Visualize the detected correspondences.

## 🗂️ Repository structure
```bash
task2/
│
├── data/                  # Sentinel-2 .SAFE scenes (not included)
│
├── weights/               # Pretrained LoFTR weights (not included)
│
├── notebooks/
│   ├── dataset_creation.ipynb     # Dataset preparation workflow
│   └── demo.ipynb                 # Pipeline demonstration
│
├── src/
│   ├── io.py                      # Reading Sentinel-2 imagery
│   ├── tiling.py                  # Image tiling and filtering
│   ├── matcher.py                 # LoFTR inference and tile ranking
│   ├── pipeline.py                # Complete matching pipeline
│   └── visualization.py           # Visualization utilities
│
├── inference.py                   # Run the complete pipeline
├── requirements.txt               # Project dependencies
└── README.md
```

# Dataset

The project uses Sentinel-2 Level-1C imagery from the public Kaggle dataset:

https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine

The assignment allows using either the official Sentinel-2 source or the provided Kaggle dataset.

The full dataset is approximately **36 GB**. For this project, only two Sentinel-2 `.SAFE` scenes are required.

The demonstration uses two Sentinel-2 scenes covering the same geographic tile (`T36UYA`) acquired in different seasons:

- `S2A_MSIL1C_20160621T084012_N0204_R064_T36UYA_20160621T084513.SAFE` Summer (June 2016)
- `S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE` Winter (February2016)

The `.SAFE` scenes are not included in this repository due to their size.

### Download dataset or scenes

These two scenes can be downloaded separately from the Kaggle dataset
https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine

After downloading, place the `.SAFE` folders inside:

```bash
data/
```

# Model weights
This project uses the pretrained **LoFTR Outdoor** model.

Weights are not included in the repository.

Download the pretrained weights:

> **[(https://drive.google.com/file/d/1IZag9Q3WLyBI_1RX0QlCC07xINbAIlRp/view?usp=sharing)]**

Place the file here:

```text
weights/
└── loftr_outdoor.ckpt
```

# Jupyter notebooks

## dataset_creation.ipynb

Explains the complete dataset preparation process.

Includes:

- Sentinel-2 dataset overview;
- RGB band construction;
- image tiling;
- valid tile filtering;
- tile statistics;
- brightness distribution.

## demo.ipynb

Demonstrates the complete image matching pipeline.

Includes:

- loading Sentinel-2 scenes;
- running the matching pipeline;
- selecting the best tile;
- visualizing LoFTR correspondences;
- reporting matching statistics.

