# AM-Mamba: Adaptive Medical Mamba for Pneumonia Classification

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-red)](.)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

> **Note**: This repository contains the implementation of our paper currently under review. Full code and pretrained models will be released upon paper acceptance.

## 📋 Overview

AM-Mamba is a novel deep learning architecture for pneumonia classification from chest X-ray (CR) images. Our method achieves **95.40% accuracy** on 4-class pneumonia classification (Normal, COVID-19, Viral Pneumonia, Lung Opacity), outperforming state-of-the-art methods including ConvNeXt V2, ResNet-50, and VMamba-T.

### Key Features

- **Dual-Branch Architecture**: Combines Mamba's global long-range modeling with enhanced Transformer's local feature extraction
- **Pathology-Specific Adaptation**: Dynamic parameter adjustment based on detected pathology patterns
- **Adaptive Scanning Mechanism**: Intelligent multi-directional scanning strategies for medical images
- **Interactive Feature Fusion**: Bidirectional feature interaction between global and local branches

## 🎯 Performance

### Main Results

| Method | FLOPs (G) ↓ | Parameters (M) ↓ | Accuracy ↑ | Balanced Accuracy ↑ | Macro-F1 ↑ |
|--------|-------------|------------------|------------|---------------------|------------|
| ConvNeXt V2 | 15.4 | 28.6 | 93.60 | 94.10 | 94.05 |
| ResNet-50 | 4.1 | 25.6 | 88.50 | 89.20 | 88.90 |
| DenseNet-121 | 2.9 | 8.0 | 89.80 | 89.20 | 88.90 |
| Twins-ViT-pcpvt | 6.8 | 22.0 | 93.31 | 93.40 | 93.35 |
| Swin-T | 4.5 | 28.3 | 91.85 | 92.10 | 91.95 |
| ViT-Base | 17.6 | 86.6 | 88.26 | 88.50 | 88.35 |
| MedMamba | 7.2 | 26.8 | 93.60 | 93.70 | 93.65 |
| VMamba-T | 5.6 | 26.1 | 93.20 | 93.30 | 93.25 |
| Vim-T | 4.9 | 25.3 | 89.30 | 89.50 | 89.40 |
| **AM-Mamba (Ours)** | **6.8** | **30.2** | **95.40** | **95.00** | **94.80** |

### Ablation Study

| Configuration | Pathology S6 | Adaptive Scanning | Enhanced Transformer | Accuracy | Gain |
|---------------|--------------|-------------------|---------------------|----------|------|
| VMamba | - | - | - | 93.20 | - |
| + Pathology S6 | ✓ | - | - | 94.00 | +0.80% |
| + Adaptive Scanning | ✓ | ✓ | - | 94.70 | +1.50% |
| + Enhanced Transformer | ✓ | ✓ | ✓ | 95.40 | +2.20% |
| **AM-Mamba (Full)** | ✓ | ✓ | ✓ | **95.40** | **+2.20%** |

## 🏗️ Architecture

AM-Mamba consists of three key innovations:

### 1. Pathology S6 Parameters
- Dynamic state-space parameters adapted to specific pathology patterns
- Progressive strength scheduling for stable training
- Medical constraint module for domain-specific regularization

### 2. Adaptive Scanning Module
- 8 intelligent scanning patterns (horizontal, vertical, diagonal, zigzag, spiral, and their reverses)
- Content-aware pattern weighting
- Automatic selection of optimal scanning strategies

### 3. Enhanced Medical Transformer
- Multi-scale feature pyramid (7×7, 14×14, 28×28 windows)
- Edge-enhanced texture extraction for CR images
- Global-local feature interaction
- Channel-spatial attention mechanism

## 📊 Dataset

We evaluate on a large-scale pneumonia dataset with 4 classes:
- **Normal**: Healthy chest X-rays
- **COVID-19**: COVID-19 pneumonia
- **Viral Pneumonia**: Non-COVID viral pneumonia
- **Lung Opacity**: Other lung abnormalities

**Dataset Statistics**:
- Total Images: 21,165
- Training Set: 16,932 (80%)
- Validation Set: 2,116 (10%)
- Test Set: 2,117 (10%)
- Image Size: 224×224 pixels
- Format: Grayscale chest X-ray (CR)

## 🚀 Quick Start

### Requirements

```bash
Python >= 3.8
PyTorch >= 2.0.0
CUDA >= 11.8 (for GPU training)
```

### Installation

```bash
# Clone the repository (will be available upon paper acceptance)
git clone https://github.com/[username]/AM-Mamba.git
cd AM-Mamba

# Install dependencies
pip install -r requirements.txt

# Install Mamba SSM (optional, for faster inference)
pip install mamba-ssm
```

### Model Variants

We provide three model sizes:

| Model | Depths | Dims | Parameters | FLOPs | Accuracy |
|-------|--------|------|------------|-------|----------|
| AM-Mamba-T | [2,2,4,2] | [96,192,384,768] | 30.2M | 6.8G | 95.40% |
| AM-Mamba-S | [2,2,8,2] | [96,192,384,768] | 44.3M | 11.5G | 95.60% |
| AM-Mamba-B | [2,2,12,2] | [128,256,512,1024] | 89.5M | 33.6G | 95.80% |

### Training

```python
# Example training script (simplified)
import torch
from model import medmamba_t_ultra

# Initialize model
model = medmamba_t_ultra()
model = model.cuda()

# Progressive S6 strength scheduling
from model import ProgressiveStrengthScheduler
scheduler = ProgressiveStrengthScheduler(
    mode='warmup',
    warmup_epochs=50,
    max_epochs=200
)

# Training loop
for epoch in range(200):
    # Update S6 strength dynamically
    s6_strength = scheduler.get_strength(epoch)
    model.update_s6_strength(s6_strength)
    
    # Your training code here
    # ...
```

### Inference

```python
import torch
from model import medmamba_t_ultra
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = medmamba_t_ultra()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()
model = model.cuda()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('chest_xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1)

# Class mapping
classes = ['Normal', 'COVID-19', 'Viral Pneumonia', 'Lung Opacity']
print(f"Prediction: {classes[pred.item()]}")
```

## 📈 Training Details

### Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (with cosine annealing)
- **Batch Size**: 16
- **Epochs**: 200
- **Weight Decay**: 0.05
- **Warmup Epochs**: 20
- **S6 Strength Schedule**: 0.1 → 1.0 (progressive)

### Data Augmentation

- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Random Affine Transform
- Color Jitter (brightness=0.2, contrast=0.2)
- Normalization (ImageNet statistics)

### Loss Function

- Primary: Cross-Entropy Loss
- Auxiliary: Label Smoothing (ε=0.1)

## 📝 Citation

If you find this work helpful, please consider citing our paper (will be updated upon acceptance):

```bibtex
@article{ammamba2024,
  title={AM-Mamba: Adaptive Medical Mamba for Pneumonia Classification},
  author={[Authors]},
  journal={[Journal Name]},
  year={2024},
  note={Under Review}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [VMamba](https://github.com/MzeroMiko/VMamba) - Vision Mamba architecture
- [Mamba](https://github.com/state-spaces/mamba) - State Space Models
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models

## 📧 Contact

For questions and discussions, please open an issue or contact:
- Email: 2023023212@stu.imust.edu.cn
- Project Page: [will be available upon acceptance]

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a preview release. Complete code, pretrained models, and detailed documentation will be made available upon paper acceptance. Stay tuned!

## 🔄 Updates

- **2025-12**: Initial repository setup
- **[TBD]**: Paper submission
- **[TBD]**: Full code release upon acceptance

