# 🧠 Biomarker-Conditioned Vision Transformers for Multimodal Alzheimer’s Disease Classification from 3D MRI and Clinical Data

## 📋 Abstract

This research introduces **BiFPN3DViT**, a novel hybrid architecture that integrates Bi-Directional Feature Pyramid Networks (BiFPN) with Vision Transformers (ViT) for 3D volumetric medical image classification. The model addresses the challenges of Alzheimer's disease detection from brain MRI scans by combining multi-scale feature extraction through BiFPN with global attention mechanisms of transformers. Our approach demonstrates state-of-the-art performance on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, achieving superior accuracy, AUC, precision, and recall compared to existing CNN-based methods for multi-class classification (CN, MCI, AD).

### Key Contributions:
- **Unified Architecture**: Seamlessly integrates BiFPN's efficient multi-scale feature fusion with ViT's global attention
- **3D Specialization**: Optimized for volumetric data with 3D convolutions and spatial attention
- **Class Imbalance Handling**: Implements Focal Loss with attention pooling for robust classification
- **Computational Efficiency**: Uses Flash Attention and stochastic depth for scalable training

---

## 🚀 Features
- 3D Convolutional feature extraction with BiFPN fusion
- Multi-head self-attention with Flash Attention support
- Focal Loss for class imbalance
- Training, validation, and test loops with rich metrics (Accuracy, AUC, Precision, Recall)
- Configurable via `config.py`
- Built-in data preprocessing pipelines
- Comprehensive evaluation with confusion matrices and classification reports

---

## 📦 Installation

### Prerequisites
- Python 3.10.15+
- CUDA-compatible GPU (recommended)

### Setup
```bash
git clone https://github.com/<your-username>/3D-BiFPN-ViT.git
cd 3D-BiFPN-ViT
pip install -r requirements.txt
```

### Data Preparation
1. Download ADNI dataset from [adni.loni.usc.edu](https://adni.loni.usc.edu/)
2. Preprocess MRI scans following the pipeline in `utils/preprocessing.py`
3. Create CSV files with paths and labels: `df_train.csv`, `df_val.csv`, `df_test.csv`

---

## 🏗️ Model Architecture

### Overview
The BiFPN3DViT architecture consists of four main components:
1. **3D Patch Embedding** - Multi-scale convolutional feature extraction
2. **BiFPN Block** - Efficient multi-scale feature fusion
3. **Transformer Encoder** - Global attention mechanism
4. **Classification Head** - Dual representation learning

### Detailed Components

#### 1. 3D Patch Embedding
- **Input**: 128×128×128×1 brain MRI volumes
- **Multi-stage CNN**: 7 convolutional blocks with increasing channels (16→32→64→128→256→512→384)
- **Output**: 5 feature maps at different scales for BiFPN fusion

#### 2. BiFPN (Bi-Directional Feature Pyramid Network)
- **Top-down pathway**: Upsampling and fusion from high-level to low-level features
- **Bottom-up pathway**: Downsampling and fusion from low-level to high-level features
- **Fusion strategy**: Weighted feature combination with 1×1×1 convolutions
- **Output**: Unified 8×8×8×384 feature representation

#### 3. Vision Transformer Encoder
- **Sequence length**: 8×8×8 = 512 patches + 1 CLS token
- **Hidden dimension**: 384
- **Attention heads**: 8
- **Layers**: 8 transformer blocks
- **Advanced features**:
  - Flash Attention for memory efficiency
  - Stochastic depth for regularization
  - Layer normalization and dropout

#### 4. Classification Head
- **Dual representation**: CLS token + attention-pooled patch features
- **Classifier**: 2-layer MLP (768→384→3) with NewGELU activation
- **Loss function**: Focal Loss (α=1.0, γ=2.0) for class imbalance

### Key Innovations
- **3D BiFPN**: Adapted BiFPN for volumetric data with 3D operations
- **Hybrid Design**: CNN feature extraction + Transformer attention
- **Memory Efficient**: Flash Attention and gradient checkpointing
- **Robust Training**: Focal loss, stochastic depth, and proper initialization

---

## 📊 Dataset Information

### Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Source**: [adni.loni.usc.edu](https://adni.loni.usc.edu/)
- **Modalities**: T1-weighted MRI scans
- **Classes**: 
  - CN (Cognitively Normal): 0
  - MCI (Mild Cognitive Impairment): 1  
  - AD (Alzheimer's Disease): 2
- **Image dimensions**: 128×128×128 voxels
- **Preprocessing**: N4 bias correction, skull stripping, normalization
- **Split**: Train/Val/Test with balanced class distribution

---

## Training Configuration
```python
# Key hyperparameters from config.py
image_size: 128
batch_size: 16
learning_rate: 1e-4
weight_decay: 1e-4
epochs: 100
focal_alpha: 1.0
focal_gamma: 2.0
```

## 🚀 Usage

### Training
```bash
# Train with default configuration
python train.py

# Custom training with arguments
python train.py --batch_size 8 --epochs 200 --learning_rate 2e-4
```

### Testing
```bash
# Evaluate trained model
python test.py --model_path weights_finalized/best_model.pth
```

### Jupyter Notebook
Explore the complete implementation in `jupyter_notebook/3D CNN-ViT_BiFPN_Final.ipynb`:
- Interactive model visualization
- Training progress monitoring
- Performance analysis and plotting

---

## 🔧 Configuration

Modify `config.py` to customize model parameters:

```python
enhanced_config = {
    "image_size": 128,
    "hidden_size": 384,          # Transformer hidden dimension
    "num_hidden_layers": 8,      # Number of transformer layers
    "num_attention_heads": 8,    # Multi-head attention heads
    "conv_channels": [16, 32, 64, 128, 256, 512],  # CNN channels
    "bifpn_channels": [64, 128, 256, 512, 384],    # BiFPN fusion channels
    "focal_alpha": 1.0,          # Focal loss alpha
    "focal_gamma": 2.0,          # Focal loss gamma
    "stochastic_depth": 0.4,     # Drop path probability
}
```

---

## 📁 Project Structure

```
3D-BiFPN-ViT/
│
├── config.py                    # Model configuration
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── models/
│   ├── __init__.py
│   └── model.py                 # BiFPN3DViT architecture
├── datasets/
│   ├── __init__.py
│   └── custom_dataset.py        # ADNI dataset loader
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py         # Data preprocessing
│   └── plotting.py              # Visualization utilities
├── jupyter_notebook/
│   └── 3D CNN-ViT_BiFPN_Final.ipynb  # Complete implementation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📜 Citation

If you use this codebase in your research, please cite:

```bibtex
@article{bifpn3dvit2024,
  title={BiFPN3DViT: A Unified 3D Vision Transformer with BiFPN for Brain MRI Classification},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## ⚠️ Disclaimer

This codebase is for research purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult with medical professionals for healthcare decisions.
