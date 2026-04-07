# 🧠 Biomarker-Conditioned Vision Transformers for Multimodal Alzheimer’s Disease Classification from 3D MRI and Clinical Data

## 📋 Abstract

Alzheimer’s disease (AD) classification from structural magnetic resonance imaging (MRI) remains an important yet challenging task, especially for distinguishing mild cognitive impairment (MCI) from both cognitively normal (CN) ageing and established AD. Multimodal approaches that combine imaging with clinical information have shown promise, but their integration strategies often underuse potentially complementary information. We developed a multimodal deep learning model that integrates 3D T1-weighted MRI with routinely collected clinical biomarkers for three-class classification of CN, MCI, and AD cases. The model was evaluated using ADNI 1.5T data with subject-wise train, validation, and test splits, ensuring that repeated scans from the same individual were not distributed across partitions. On the held-out test set, the model achieved 95.68\% accuracy, 95.39\% macro precision, 94.65\% macro recall, and 95.01\% macro F1-score, exceeding the performance of reproduced comparison methods evaluated under the same protocol. In addition, prediction uncertainty was higher for misclassified cases, and selective rejection of the most uncertain cases improved retained-set accuracy to 98.31\% at 85.03\% coverage. The results indicate that integrating structural MRI with clinical biomarkers can improve AD classification performance and provide confidence information that may be useful for decision support in clinical research settings.

### Key Contributions:


---

## 🚀 Features


---

## 📦 Installation

### Prerequisites
- Python 3.10.15+
- CUDA-compatible GPU (recommended)

### Setup
```bash
git clone https://github.com/<your-username>/multimodal-ad-classification-vit.git
cd multimodal-ad-classification-vit
pip install -r requirements.txt
```

### Data Preparation
1. Download ADNI dataset from [adni.loni.usc.edu](https://adni.loni.usc.edu/)
2. Preprocess MRI scans following the pipeline in `utils/preprocessing.py`.
- Skull Stripping (HD-BET)
- Cropping to get only Brain Regions (remove black sides).
3. Create CSV files with paths and labels: `df_train.csv`, `df_val.csv`, `df_test.csv`

---

## 🏗️ Model Architecture

### Overview


### Detailed Components




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
- **Preprocessing**: N4 bias correction, skull stripping, normalisation
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
    "bifpn_channels": [64, 128, 256, 512, 384],    # fusion channels
    "focal_alpha": 1.0,          # Focal loss alpha
    "focal_gamma": 2.0,          # Focal loss gamma
    "stochastic_depth": 0.4,     # Drop path probability
}
```

---

## 📁 Project Structure

```
3D--ViT/
│
├── config.py                    # Model configuration
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── models/
│   ├── __init__.py
│   └── model.py                 #  architecture
├── datasets/
│   ├── __init__.py
│   └── custom_dataset.py        # ADNI dataset loader
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py         # Data preprocessing
│   └── plotting.py              # Visualization utilities
├── jupyter_notebook/
│   └── 3D CNN-ViT_Final.ipynb  # Complete implementation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📜 Citation

If you use this codebase in your research, please cite:


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
