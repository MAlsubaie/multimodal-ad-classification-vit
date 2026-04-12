# 🧠 Biomarker-Conditioned Vision Transformers for Multimodal Alzheimer’s Disease Classification from 3D MRI and Clinical Data

## 📋 Abstract

Alzheimer’s disease (AD) classification from structural magnetic resonance imaging (MRI) remains an important yet challenging task, especially for distinguishing mild cognitive impairment (MCI) from both cognitively normal (CN) ageing and established AD. Multimodal approaches that combine imaging with clinical information have shown promise, but their integration strategies often underuse potentially complementary information. We developed a multimodal deep learning model that integrates 3D T1-weighted MRI with routinely collected clinical biomarkers for three-class classification of CN, MCI, and AD cases. The model was evaluated using ADNI 1.5T data with subject-wise train, validation, and test splits, ensuring that repeated scans from the same individual were not distributed across partitions. On the held-out test set, the model achieved 95.68\% accuracy, 95.39\% macro precision, 94.65\% macro recall, and 95.01\% macro F1-score, exceeding the performance of reproduced comparison methods evaluated under the same protocol. In addition, prediction uncertainty was higher for misclassified cases, and selective rejection of the most uncertain cases improved retained-set accuracy to 98.31\% at 85.03\% coverage. The results indicate that integrating structural MRI with clinical biomarkers can improve AD classification performance and provide confidence information that may be useful for decision support in clinical research settings.
---

## 🚀 Features
- 3D Patch Embedding for volumetric MRI data.
- Biomarker Encoder with coordinate learning for spatial reasoning.
- Multi-head self-attention integrated with Deformable Biomarker Attention.
- Two-stage gating mechanism to blend standard and biomarker-conditioned attention.
- CrossEntropy loss for multi-class optimization.
- Uncertainty-aware prediction testing via MC Dropout (`predict_with_uncertainty`).
- Configurable via `config.py`.

---

## 📦 Installation

### Prerequisites
- Python 3.10.15+
- CUDA-compatible GPU (recommended)

### Setup
```bash
git clone https://github.com/<your-username>/Biomarker-Conditioned-ViT.git
cd Biomarker-Conditioned-ViT
pip install -r requirements.txt
```

### Data Preparation
1. Download the ADNI dataset from [adni.loni.usc.edu](https://adni.loni.usc.edu/).
2. Preprocess MRI scans following the established SDB80 and brainmask pipeline.
3. Create CSV files with paths, labels, and relevant biomarker columns (`Age_x`, `MMSE Total Score`, `GDSCALE Total Score`, `Global CDR`, `FAQ Total Score`).
4. Ensure files are named appropriately (`df_train.csv`, `df_val.csv`, `df_test.csv`) and map to your training scripts.

---

## 🏗️ Model Architecture

### Overview
The BiomarkerConditionedViT architecture consists of four main components:
1. **3D Patch Embedding** - Linear projection of 128³ MRI volumes into 16³ voxel patches.
2. **Biomarker Encoder** - Transforms continuous biomarkers into modulated tokens, spatial maps, and dynamic DBA offsets.
3. **Deformable Biomarker Attention** - Samples spatial features dynamically directed by learned base coordinates and biomarker-dependent offsets.
4. **Biomarker-Conditioned Transformer Blocks** - Fuses standard self-attention, biomarker-conditioned attention, and DBA using a two-stage gating mechanism.

### Key Innovations
- **DBA (Deformable Biomarker Attention)**: Employs `F.grid_sample` logic directly inside transformer blocks to fetch spatial features based on continuous biomarkers.
- **Confidence Fusion**: Predictions are weighed against a simultaneously learned confidence score, allowing the model to estimate uncertainty on varying subsets of modalities.

---

## 📊 Dataset Information

### Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Classes**: 
  - CN (Cognitively Normal): 0
  - MCI (Mild Cognitive Impairment): 1  
  - AD (Alzheimer's Disease): 2
- **Image dimensions**: 128×128×128 voxels (1 channel)
- **Biomarkers Required**: 5 numerical features (default: Age, MMSE, GDSCALE, Global CDR, FAQ).

---

## 🚀 Usage

### Training
```bash
# Train with default configuration from config.py
python train.py

# Custom training with arguments
python train.py --batch_size 16 --epochs 100 --learning_rate 1e-4
```

### Testing
```bash
# Evaluate trained model
python test.py --weights_path weights_finalized/best_model.pth
```

### Checking Model Statistics & Sandbox
Run the model codebase directly to verify the forward passes and observe dummy visualizations:
```bash
python models/model.py
```

---

## 🔧 Configuration

Modify `config.py` to customize model parameters:

```python
enhanced_config = {
    "image_size": 128,
    "patch_size": 16,
    "num_channels": 1,
    "num_classes": 3,
    "num_biomarkers": 5,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "conditioning_layers": [4, 5, 6, 7, 8, 9, 10, 11]
}
```

---

## 📁 Project Structure

```
Biomarker-Conditioned-ViT/
│
├── config.py                    # Model configuration
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── models/
│   ├── __init__.py
│   └── model.py                 # Core DBA-ViT architecture
├── datasets/
│   ├── __init__.py
│   └── custom_dataset.py        # ADNI multimodal dataset loader
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

---

## ⚠️ Disclaimer

This codebase is for research purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult with medical professionals for healthcare decisions.
