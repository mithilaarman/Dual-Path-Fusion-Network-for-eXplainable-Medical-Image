# P2FNet: A Dual-Path Fusion Network for eXplainable Medical Image Classification

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-green)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM-purple)

Official implementation of **P2FNet**, a lightweight dual-path fusion architecture for explainable medical image classification.

P2FNet combines complementary **local fine-grained features** and **global semantic information** through an efficient late-fusion strategy. The framework is evaluated across six public medical imaging datasets representing multiple imaging modalities.

---

## 🔍 Overview

Medical image classification requires models to capture both:

* subtle and localized pathological patterns, and
* broader semantic and contextual information.

Convolutional neural networks are effective at learning local patterns, while global representations are important for understanding long-range contextual relationships.

**P2FNet** addresses this problem using two parallel feature extraction pathways:

1. **Global Feature Extractor (GFE)**
2. **Local Feature Extractor (LFE)**

The representations produced by the two branches are combined through **channel-wise late fusion** before classification.

---

## 🧠 Architecture

The proposed architecture consists of two complementary branches.

### Global Feature Extractor

The global branch uses a pretrained **MobileNetV2** backbone to capture high-level semantic information and global contextual features.

### Local Feature Extractor

The local branch is designed to capture fine-grained pathological patterns using:

* Depthwise Separable Convolutions
* Squeeze-and-Excitation (SE) blocks
* Residual connections

### Late Feature Fusion

Both branches produce feature representations that are passed through **Global Average Pooling**.

The resulting vectors are concatenated using channel-wise fusion:

```text
                    Input Image
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
     Global Feature             Local Feature
       Extractor                  Extractor
     (MobileNetV2)         (Depthwise Conv + SE)
             │                       │
             ▼                       ▼
     Global Average             Global Average
        Pooling                    Pooling
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                Channel-wise Fusion
                         │
                         ▼
                 Fully Connected
                    Classifier
                         │
                         ▼
                    Prediction
```

The architecture is designed to retain the distinct strengths of both feature streams while maintaining low computational complexity.

---

## 🩺 Medical Imaging Datasets

P2FNet is evaluated on six public datasets spanning several medical imaging modalities.

| Dataset            | Imaging Modality    | Classes | Images |
| ------------------ | ------------------- | ------: | -----: |
| SIPaKMeD           | Optical Microscopy  |       5 |  4,049 |
| SARS-COV-2 CT-Scan | Computed Tomography |       2 |  2,482 |
| APTOS              | Fundus Photography  |       2 |  3,662 |
| Kvasir             | Endoscopy           |       8 |  4,000 |
| Malaria            | Optical Microscopy  |       2 | 27,558 |
| HAM10000           | Dermoscopy          |       7 | 10,015 |

This multi-dataset evaluation is used to assess the generalizability of the proposed framework across different medical imaging domains.

---

## ⚙️ Preprocessing

All images are resized to:

```text
224 × 224
```

For pretrained architectures, ImageNet normalization is used:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

For the custom P2FNet training pipeline, image values are rescaled to the `[0, 1]` range.

### Data Augmentation

The training pipeline incorporates several augmentation techniques, including:

* Horizontal flipping
* Vertical flipping
* Random rotation
* Affine transformations
* Perspective distortion
* Random inversion
* Color jitter
* Grayscale conversion

These transformations are used to improve robustness and reduce overfitting.

---

## 🏋️ Training Configuration

The proposed model is trained using the following configuration:

| Setting               | Value                  |
| --------------------- | ---------------------- |
| Epochs                | 100                    |
| Optimizer             | AdamW                  |
| Initial Learning Rate | 0.0001                 |
| Batch Size            | 16                     |
| Weight Decay          | `1 × 10⁻⁴`             |
| Scheduler             | ReduceLROnPlateau      |
| Scheduler Patience    | 5                      |
| Early Stopping        | 15 epochs              |
| Binary Loss           | BCEWithLogitsLoss      |
| Multi-class Loss      | CrossEntropyLoss       |
| Model Selection       | Lowest Validation Loss |

Experiments were conducted using an **NVIDIA Tesla P100 GPU with 16 GB VRAM**.

---

## 🧪 Baseline Models

P2FNet is compared against representative CNN and Transformer architectures.

### CNN Models

* VGG16
* EfficientNet-B4
* MobileNetV3
* ConvNeXt-Tiny

### Transformer Models

* ViT-Tiny
* Swin-Tiny
* DeiT-Base

These models are fine-tuned using ImageNet pretrained weights.

---

## 📊 Evaluation Metrics

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* AUC

The evaluation additionally includes:

* training and validation curves,
* confusion matrices,
* ROC curves,
* model component ablations,
* fusion strategy ablations, and
* Grad-CAM explainability analysis.

---

## 🏆 Main Results

P2FNet achieves the strongest reported accuracy across all six evaluated datasets.

| Dataset        |   Accuracy |
| -------------- | ---------: |
| **APTOS**      | **98.64%** |
| **SARS-COV-2** | **98.79%** |
| **SIPaKMeD**   | **97.79%** |
| **Malaria**    | **97.02%** |
| **HAM10000**   | **88.89%** |
| **Kvasir**     | **90.30%** |

The results demonstrate that combining local and global representations can provide consistent performance across heterogeneous medical imaging modalities.

---

## ⚡ Lightweight Design

Despite using two parallel feature extraction pathways, P2FNet maintains a relatively small computational footprint:

```text
Parameters : 3.7M
FLOPs      : 3.62G
```

The architecture is designed to provide a balance between:

**classification performance × computational efficiency × generalizability**

---

## 🔬 Ablation Study

Component-level experiments on the APTOS dataset show the complementary contribution of the two feature streams.

| Architecture             |   Accuracy |         F1 |
| ------------------------ | ---------: | ---------: |
| Local Stream Only        |     97.82% |     97.81% |
| Global Stream Only       |     98.09% |     98.07% |
| Early Fusion             |     97.45% |     97.43% |
| Mid-Level Fusion         |     97.83% |     97.80% |
| Multi-Scale Fusion       |     98.30% |     98.29% |
| **Late Fusion (P2FNet)** | **98.64%** | **98.62%** |

The results indicate that the local and global streams learn complementary representations and that **late fusion provides the strongest performance** among the evaluated fusion strategies.

---

## 🔥 Explainable AI

P2FNet incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the regions contributing to model predictions.

Grad-CAM analysis is performed across the evaluated medical imaging modalities, including:

* retinal fundus images,
* CT scans,
* cervical cell microscopy,
* gastrointestinal endoscopy,
* dermoscopic skin images, and
* malaria microscopy images.

The visualizations are used to examine whether the model focuses on diagnostically relevant image regions.

---

## 💡 Key Contributions

* A lightweight **dual-path architecture** combining local and global representations.
* Efficient local feature extraction using depthwise separable convolutions and SE attention.
* Pretrained global semantic feature extraction using MobileNetV2.
* Simple and effective **channel-wise late fusion**.
* Evaluation across **six diverse medical imaging datasets**.
* Comparison against modern CNN and Transformer baselines.
* Grad-CAM-based explainability analysis.
* Component and fusion-strategy ablation experiments.
* Only **3.7M parameters** with **3.62G FLOPs**.

---

## 🛠️ Environment

The experiments described in the paper use:

```text
Python       3.11.13
PyTorch      2.6.0
TorchVision  0.21.0
TIMM         1.0.15
```

---

## 📄 Paper

**P2FNet: A Dual-Path Fusion Network for eXplainable Medical Image Classification**

**Authors:**
Md. Mahid Arfan Rahat, Mithila Arman, Abid Haider, Isha Das, and Mehedi Hasan

---

## 📚 Citation

If this work is useful for your research, please cite the corresponding paper.

```bibtex
@article{p2fnet2026,
  title   = {P2FNet: A Dual-Path Fusion Network for eXplainable Medical Image Classification},
  author  = {Rahat, Md. Mahid Arfan and Arman, Mithila and Haider, Abid and Das, Isha and Hasan, Mehedi},
  year    = {2026}
}
```

> Journal, volume, pages, and DOI can be added to the BibTeX entry once the final publication metadata is available.

---

## 🔮 Future Work

Future extensions of P2FNet may investigate more advanced adaptive feature-fusion mechanisms to further improve the interaction between local and global representations while preserving computational efficiency.
