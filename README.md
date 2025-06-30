# NIR-VIS Face Recognition with Triplet Loss

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Google%20Colab-yellow.svg)

## Overview

This repository contains a Jupyter Notebook (`final_code.ipynb`) implementing a face recognition system for Near-Infrared (NIR) and Visible (VIS) spectrum images using a triplet loss approach with a ResNet50 backbone. The system learns embeddings to minimize the distance between images of the same identity (across NIR and VIS spectra) while maximizing the distance for different identities.

The code is designed to run on Google Colab with GPU support for optimal performance.

## Features

- ✨ **Cross-Spectral Face Recognition**: Works with both NIR and VIS images
- 🧠 **ResNet50 Backbone**: Pre-trained on ImageNet for robust feature extraction
- 📊 **Triplet Loss Training**: Semi-hard negative mining for effective embedding learning
- 📈 **Comprehensive Evaluation**: CMC curves and confusion matrices
- ⚡ **GPU Acceleration**: Optimized for Google Colab GPU runtime
- 🔄 **Mixed Precision Training**: Faster training with reduced memory usage

## Dataset

The **NIR-VIS-2.0** dataset is used, containing paired NIR and VIS face images organized by identity. Images are preprocessed to 128×128 pixels and normalized.

### Directory Structure
```
/content/dataset/s1/
├── VIS_128x128/    # Visible spectrum images
└── NIR_128x128/    # Near-infrared images
```

**Note**: The dataset should be stored in Google Drive and mounted in Colab at `/content/drive/MyDrive/NIR-VIS-2.0.zip`.

## Prerequisites

### Software Requirements
- Python 3.11+
- TensorFlow 2.18.0 (with CUDA and cuDNN for GPU support)
- NumPy
- OpenCV (cv2)
- Scikit-learn
- Matplotlib
- Seaborn

### Hardware Requirements
- Google Colab with GPU runtime (e.g., Tesla T4)
- Minimum 4GB GPU memory recommended

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up Google Colab
1. Open Google Colab
2. Change runtime type to **GPU** (Runtime → Change runtime type → Hardware accelerator → GPU)

### 3. Install Required Packages
```python
!pip install tensorflow opencv-python-headless scikit-learn matplotlib seaborn
```

### 4. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Extract Dataset
```python
from zipfile import ZipFile

zip_file_path = '/content/drive/MyDrive/NIR-VIS-2.0.zip'
extract_path = '/content/dataset/'

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

## Usage

### Running the Notebook

1. Upload `final_code.ipynb` to Google Colab
2. Ensure dataset paths (`vis_dirs` and `nir_dirs`) match your directory structure
3. Run all cells sequentially

### Key Components

| Component | Description |
|-----------|-------------|
| **Data Loading** | Loads paired NIR and VIS images, ensuring common identities |
| **Preprocessing** | Resizes images to 128×128, normalizes pixel values, filters classes with <10 samples |
| **Model Architecture** | ResNet50 backbone with 128-dimensional embedding layer |
| **Training** | Triplet loss with semi-hard negative mining, Adam optimizer, mixed precision |
| **Evaluation** | CMC curves and confusion matrices for both spectra |

### Model Architecture

```
Input (128x128x3) → ResNet50 (pretrained) → Global Average Pooling → Dense(128) → L2 Normalization
```

## Outputs

The system generates several evaluation metrics and visualizations:

- 📊 **CMC Curves**: Identification rates at different ranks for VIS and NIR spectra
- 🔥 **Confusion Matrices**: Classification performance visualization
- 📈 **Training Loss Plot**: Training and validation triplet loss over epochs
- ⏰ **Timestamped Results**: All outputs include generation timestamps

### Sample Results
- **Fig. 4**: CMC Curve for VIS spectrum
- **Fig. 5**: CMC Curve for NIR spectrum

## Configuration

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 32 | Training batch size |
| `num_triplets` | 1000 | Number of triplets per epoch |
| `min_samples` | 10 | Minimum samples per class |
| `epochs` | 50 | Training epochs |
| `embedding_dim` | 128 | Embedding dimension |

### Customization

Modify these parameters in the notebook to suit your dataset or performance needs:

```python
# Training configuration
BATCH_SIZE = 32
NUM_TRIPLETS = 1000
MIN_SAMPLES = 10
EPOCHS = 50
EMBEDDING_DIM = 128
```

## Repository Structure

```
.
├── final_code.ipynb          # Main Jupyter Notebook
├── README.md                 # This file
└── requirements.txt          # Python dependencies (optional)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Empty Dataset Error** | Verify dataset paths and ensure common identities exist in NIR and VIS directories |
| **GPU Issues** | Confirm Colab is set to GPU runtime and CUDA/cuDNN are configured correctly |
| **Memory Errors** | Reduce `batch_size` or `num_triplets` if GPU memory is insufficient |
| **Import Errors** | Ensure all required packages are installed using the provided pip commands |

### Performance Tips

- Use GPU runtime for faster training (10-20x speedup)
- Enable mixed precision training to reduce memory usage
- Monitor GPU memory usage during training
- Consider reducing image resolution if memory is limited

## Dataset Licensing

⚠️ **Important**: The NIR-VIS-2.0 dataset is not included in this repository due to licensing restrictions. Please:

1. Obtain the dataset separately from the official source
2. Ensure compliance with the dataset's terms of use
3. Follow all licensing requirements for academic/commercial use

## Contributing

Contributions are welcome! Here's how you can help:

1. 🐛 **Bug Reports**: Open an issue with detailed reproduction steps
2. ✨ **Feature Requests**: Suggest new features or improvements
3. 🔧 **Pull Requests**: Submit bug fixes or enhancements
4. 📖 **Documentation**: Help improve documentation and examples

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- 🙏 Built using TensorFlow and ResNet50 pretrained weights from ImageNet
- 🚀 Designed for Google Colab's GPU environment
- 📚 Inspired by state-of-the-art cross-spectral face recognition research

## Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{nir-vis-face-recognition,
  title={NIR-VIS Face Recognition with Triplet Loss},
  author={RDX463},
  year={2025},
  url={https://github.com/RDX463/face_recognition}
}
```

---

<div align="center">
Made with ❤️ for cross-spectral face recognition research
</div>
