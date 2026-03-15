# 🛰 Satellite-Based Urban Growth Analysis — Vijayawada

A complete **deep learning pipeline** for satellite-based urban growth analysis of **Vijayawada, Andhra Pradesh, India**, using Sentinel-2 imagery from Google Earth Engine.

---

## 🏙 Overview

This system analyses urban expansion in Vijayawada between **2020 and 2023** using multi-spectral Sentinel-2 satellite imagery and deep learning models for semantic segmentation and change detection.

| Category | Details |
|----------|---------|
| **City** | Vijayawada, Andhra Pradesh, India |
| **Coordinates** | 16.5062°N, 80.6480°E |
| **Satellite Data** | Sentinel-2 MSI (10m resolution) |
| **Analysis Period** | 2020 – 2023 |
| **Classes** | Non-Urban · Semi-Urban · Urban |
| **Models** | U-Net · ResNet50 · EfficientNet · Siamese Change Detector |

---

## 📁 Project Structure

```
satellite-urban-growth-analysis/
├── config/
│   ├── __init__.py
│   └── config.yaml            ← All project settings
├── data/
│   ├── raw/                   ← Downloaded GeoTIFF files
│   ├── processed/             ← .npy patch datasets
│   └── exports/               ← GEE export outputs
├── gee_scripts/
│   ├── sentinel2_extract.py   ← GEE data extraction
│   ├── spectral_indices.py    ← NDVI, NDBI, NDMI etc.
│   └── change_detection.py    ← Multi-temporal analysis
├── src/
│   ├── data/
│   │   ├── dataset.py         ← PyTorch Dataset classes
│   │   ├── dataloader.py      ← DataLoader factory
│   │   ├── augmentation.py    ← Albumentations transforms
│   │   └── preprocessing.py   ← GeoTIFF I/O, patches, PCA
│   ├── models/
│   │   ├── unet.py            ← U-Net segmentation
│   │   ├── resnet.py          ← ResNet50 classifier/segmenter
│   │   ├── efficientnet.py    ← EfficientNet models
│   │   ├── temporal_model.py  ← Siamese + LSTM change detection
│   │   └── losses.py          ← Dice, Focal, Tversky, Combined
│   ├── training/
│   │   ├── trainer.py         ← Full training loop + AMP
│   │   ├── metrics.py         ← IoU, Dice, F1, Confusion Matrix
│   │   └── callbacks.py       ← Early stopping, checkpointing
│   ├── inference/
│   │   ├── predictor.py       ← Patch & sliding-window inference
│   │   └── postprocessing.py  ← Morphological ops, GeoTIFF export
│   └── visualization/
│       ├── maps.py            ← Folium / Geemap interactive maps
│       ├── plots.py           ← Matplotlib plots
│       └── reports.py         ← HTML/Markdown report generation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_inference.ipynb
│   └── 05_analysis.ipynb
├── scripts/
│   ├── download_data.py       ← GEE data download
│   ├── preprocess_data.py     ← Patch extraction & splitting
│   ├── train_model.py         ← Model training CLI
│   ├── evaluate_model.py      ← Evaluation CLI
│   └── generate_maps.py       ← Map & report generation
├── checkpoints/               ← Model checkpoints (.pth)
├── logs/                      ← TensorBoard logs
├── results/
│   ├── predictions/           ← GeoTIFF output maps
│   ├── visualizations/        ← PNG/HTML maps
│   └── reports/               ← HTML/Markdown reports
├── requirements.txt
├── setup.py
└── .env.example
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Sowmya-n30/satellite-urban-growth-analysis.git
cd satellite-urban-growth-analysis

pip install -r requirements.txt
# or install as a package:
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Download Vijayawada Data (Google Earth Engine)

```bash
# Authenticate with GEE first
python -c "import ee; ee.Authenticate()"

# Download Sentinel-2 data for all years
python scripts/download_data.py --years 2020 2021 2022 2023 --folder vijayawada_gee
```

### 4. Preprocess Data

```bash
python scripts/preprocess_data.py \
    --input data/raw/ \
    --output data/processed/ \
    --patch-size 256 \
    --stride 128
```

### 5. Train a Model

```bash
# Train U-Net (recommended)
python scripts/train_model.py \
    --model unet \
    --data data/processed/ \
    --epochs 100 \
    --batch-size 16

# Train ResNet50 segmenter
python scripts/train_model.py --model resnet50 --data data/processed/

# Train EfficientNet
python scripts/train_model.py --model efficientnet --data data/processed/
```

### 6. Evaluate

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/unet_vijayawada/best_model.pth \
    --data data/processed/ \
    --model unet
```

### 7. Generate Maps & Reports

```bash
python scripts/generate_maps.py \
    --checkpoint checkpoints/unet_vijayawada/best_model.pth \
    --image-t1 data/raw/vijayawada_S2_2020.tif \
    --image-t2 data/raw/vijayawada_S2_2023.tif \
    --output results/
```

---

## 🧠 Deep Learning Models

### U-Net (Semantic Segmentation)
- Multi-scale encoder-decoder with skip connections
- Supports 10–15 input channels (Sentinel-2 + spectral indices)
- Output: Per-pixel classification (Non-Urban / Semi-Urban / Urban)

### ResNet50 (Classification & Segmentation)
- ImageNet pre-trained backbone, fine-tuned for multi-spectral input
- Feature-pyramid decoder for segmentation
- Output: Urban growth classification

### EfficientNet (Lightweight Inference)
- EfficientNet-B0 encoder with FPN decoder
- Scalable for mobile/edge deployment
- Output: Real-time urban classification

### Siamese Change Detector (Temporal Analysis)
- Shared encoder compares t1 and t2 images
- Bottleneck fusion for change maps
- Output: Binary change mask (new urban / unchanged)

---

## 📊 Spectral Indices

| Index | Formula | Use |
|-------|---------|-----|
| **NDVI** | (B8-B4)/(B8+B4) | Vegetation detection |
| **NDBI** | (B11-B8)/(B11+B8) | Built-up area detection |
| **NDMI** | (B8-B11)/(B8+B11) | Moisture index |
| **MNDWI** | (B3-B11)/(B3+B11) | Water body detection |
| **EVI** | 2.5*(B8-B4)/(B8+6B4-7.5B2+1) | Enhanced vegetation |

---

## 📈 Training Features

- ✅ Mixed precision training (AMP)
- ✅ CosineAnnealing LR scheduler
- ✅ Early stopping with configurable patience
- ✅ Best model checkpoint saving
- ✅ TensorBoard logging
- ✅ Class-weighted loss for imbalanced datasets
- ✅ Combined loss: Dice + CrossEntropy + Focal
- ✅ Metrics: IoU, Dice, F1, Precision, Recall, OA

---

## 🗺 GEE Data for Vijayawada

```python
import ee
ee.Authenticate()
ee.Initialize()

# Vijayawada ROI
roi = ee.Geometry.Point([80.6480, 16.5062]).buffer(30000)

# Annual Sentinel-2 composites (2020-2023)
for year in [2020, 2021, 2022, 2023]:
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median().clip(roi))
    
    # NDBI for built-up detection
    ndbi = s2.normalizedDifference(['B11', 'B8'])
    
    # Export to Drive
    ee.batch.Export.image.toDrive(
        image=s2, description=f'vijayawada_{year}',
        folder='vijayawada_gee', scale=10, region=roi
    ).start()
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis of Sentinel-2 imagery |
| `02_data_preparation.ipynb` | Patch extraction and dataset preparation |
| `03_model_training.ipynb` | Model training with metrics visualization |
| `04_inference.ipynb` | Inference on full Vijayawada scene |
| `05_analysis.ipynb` | Urban growth analysis and reporting |

---

## 📦 Best Free Datasets

| Dataset | Resolution | Size | Pre-labeled | Link |
|---------|-----------|------|------------|------|
| **UC Merced** | 0.3m | 500 MB | ✅ | [Download](http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip) |
| **NWPU RESISC45** | Various | 1.9 GB | ✅ | [GitHub](https://github.com/YushiChen98/RemoteSensing-datasets) |
| **Sentinel-2 (GEE)** | 10m | Unlimited | ❌ | [GEE](https://earthengine.google.com/) |
| **SpaceNet** | 0.5-5m | 5 TB | ✅ | [AWS](https://registry.opendata.aws/spacenet/) |
| **BigEarthNet** | 10m | 665 GB | ✅ | [Download](http://bigearth.net/) |

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
