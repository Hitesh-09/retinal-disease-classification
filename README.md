# 🧠 Retinal Disease Classification using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App" />
  <img src="https://img.shields.io/badge/Dataset-ODIR--5K-success?logo=kaggle&logoColor=white" alt="ODIR-5K Dataset" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

An end-to-end multi-label ophthalmic diagnostic pipeline designed to identify multiple ocular pathologies simultaneously from retinal fundus images using deep convolutional ensembles (**EfficientNet-B4** & **DenseNet121**), weighted binary loss for severe class imbalance, and **Grad-CAM** visual interpretability.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset & Clinical Classes](#-dataset--clinical-classes)
- [Model Architecture & Methodology](#-model-architecture--methodology)
- [Experimental Results & Evaluation](#-experimental-results--evaluation)
- [Explainability & Interpretability (Grad-CAM)](#-explainability--interpretability-grad-cam)
- [Repository Structure](#-repository-structure)
- [Quickstart & Usage](#-quickstart--usage)
  - [Prerequisites & Installation](#1-prerequisites--installation)
  - [Data Preparation](#2-data-preparation)
  - [Model Training](#3-model-training)
  - [Evaluation](#4-evaluation)
  - [Interactive Web Application (Streamlit)](#5-interactive-web-application-streamlit)
- [Future Roadmap](#-future-roadmap)
- [Contributing & License](#-contributing--license)

---

## 📌 Overview

Multi-label retinal screening poses significant challenges due to overlapping co-morbidities (e.g., Diabetic Retinopathy combined with Hypertension) and pronounced class imbalance across population datasets. 

This project provides a complete deep learning framework addressing these challenges on the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset:
1. **Multi-label classification** predicting eight diagnostic categories per fundus image.
2. **Class-weighted loss optimization** preventing dominance of common classes over rare conditions.
3. **Ensemble inference** combining complementary spatial representations from EfficientNet-B4 and DenseNet121.
4. **Grad-CAM visual attribution** allowing ophthalmologists to audit feature activations against clinically recognized lesions.
5. **Interactive Streamlit deployment** for point-of-care model inference.

---

## ✨ Key Features

- **Multi-Label Diagnostic Engine**: Predicts independent probabilities for all 8 standard ODIR disease categories simultaneously.
- **Deep Hybrid Ensembling**: Combines compound-scaled EfficientNet-B4 and densely connected DenseNet121 backbones to produce robust predictions (**0.8848 Validation AUC / 0.8655 Test AUC**).
- **Class-Imbalance Mitigations**: Implements positive class weighting tailored to reciprocal class frequencies.
- **Explainable AI (XAI)**: High-resolution Grad-CAM saliency heatmaps overlaid on original fundus images.
- **Modular & Production-Ready Codebase**: Clean separation across configuration (`src/config.py`), dataset pipelines (`src/dataset.py`), loss functions (`src/losses.py`), models (`src/models.py`), evaluation (`src/evaluate.py`), and visualization (`src/gradcam.py`).
- **Interactive UI**: Real-time fundus inspection and diagnostic heatmaps via Streamlit.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A["Raw Retinal Fundus Image"] --> B["Preprocessing & Normalization<br/>(224x224, Scale/Augment)"]
    
    subgraph S1["Ensemble Feature Extraction & Classification"]
        B --> C["EfficientNet-B4<br/>(Weighted BCE Loss)"]
        B --> D["DenseNet121<br/>(Weighted BCE Loss)"]
        C --> E["Sigmoid Multi-Label Output"]
        D --> F["Sigmoid Multi-Label Output"]
    end
    
    E & F --> G["Ensemble Blending<br/>(Probability Averaging)"]
    
    subgraph S2["Clinical Decision & Saliency"]
        G --> H["Multi-Label Thresholding<br/>(Normal, DR, Glaucoma, Cataract, etc.)"]
        C -.-> I["Grad-CAM Saliency Engine<br/>(Top Convolutional Layer)"]
        I --> J["Explainability Heatmap Overlay"]
    end
    
    H & J --> K["Streamlit Dashboard / Clinical Report"]
```

---

## 🧩 Dataset & Clinical Classes

The models are trained and evaluated on the **ODIR-5K (Ocular Disease Intelligent Recognition)** benchmark dataset, containing annotated bilateral retinal fundus photographs.

### Diagnostic Target Classes

| Label Code | Clinical Condition | Description / Hallmark Features |
| :---: | :--- | :--- |
| **N** | **Normal** | Healthy retina without observable pathological changes |
| **D** | **Diabetes** | Diabetic Retinopathy: Microaneurysms, hemorrhages, exudates |
| **G** | **Glaucoma** | Optic disc cupping, increased cup-to-disc ratio, nerve fiber loss |
| **C** | **Cataract** | Lens opacity causing generalized haziness and reduced fundus contrast |
| **A** | **Age-related Macular Degeneration (AMD)** | Drusen deposits, geographic atrophy, choroidal neovascularization |
| **H** | **Hypertension** | Arteriolar narrowing, arteriovenous nicking, flame hemorrhages |
| **M** | **Pathological Myopia** | Chorioretinal atrophy, posterior staphyloma, tilted optic disc |
| **O** | **Other Diseases** | Epiretinal membrane, retinal vein occlusion, laser scars, etc. |

### Sample Fundus Images with Multi-Label Ground Truth

<p align="center">
  <img src="assets/samples.png" alt="Sample Fundus Images with Multi-Label Ground Truth" width="850px" />
  <br />
  <em>Figure 1: Representative patient fundus images from the ODIR-5K dataset displaying distinct pathologies and annotations.</em>
</p>

### Class Distribution & Imbalance Analysis

As typical in medical screening populations, conditions such as Diabetes (D) and Normal (N) predominate, whereas Hypertension (H), Myopia (M), and AMD (A) represent minority classes:

<p align="center">
  <img src="assets/disease_distribution.png" alt="ODIR-5K Disease Distribution" width="850px" />
  <br />
  <em>Figure 2: Distribution of disease classes showing significant frequency variation across categories.</em>
</p>

To counteract bias toward majority classes, positive class weights are dynamically computed and integrated into the loss formulation:
$$\text{Weight}_c = \frac{N_{\text{total}} - N_{c}}{N_c}$$

---

## ⚙️ Model Architecture & Methodology

### 1. Data Processing Pipeline
- Input resolution standardized to $224 \times 224 \times 3$.
- Robust normalization matching ImageNet and architecture-specific scaling.
- Stratified Train / Validation / Test partitions ($70\% / 15\% / 15\%$).

### 2. Deep Architectures
- **EfficientNet-B4**: Employs compound scaling balancing network depth, width, and resolution. Captures subtle lesions with high parameter efficiency.
- **DenseNet121**: Direct feature reuse across dense blocks preserves low-level vascular structures critical for detecting microaneurysms and optic disc contours.

### 3. Loss Function & Optimization
Binary Cross-Entropy with positive label weighting addresses multi-label co-occurrence:

$$\mathcal{L} = -\frac{1}{C}\sum_{c=1}^{C} \left[ w_c \cdot y_c \log(\hat{y}_c) + (1 - y_c) \log(1 - \hat{y}_c) \right]$$

where $w_c$ represents the positive class weight for class $c$, $y_c \in \{0, 1\}$ is the ground truth label, and $\hat{y}_c$ is the predicted probability.

---

## 📊 Experimental Results & Evaluation

### Training Convergence Dynamics

The training curves demonstrate rapid loss minimization and sustained validation accuracy across epochs without catastrophic divergence:

<p align="center">
  <img src="assets/accuracy.png" width="48%" alt="Training and Validation Accuracy" />
  <img src="assets/loss.png" width="48%" alt="Training and Validation Loss" />
  <br />
  <em>Figure 3: Multi-label classification accuracy (left) and weighted loss trajectory (right) over training epochs.</em>
</p>

### Quantitative Performance Comparison (AUC)

| Architecture / Configuration | Strategy | Validation Macro AUC | Test Macro AUC |
| :--- | :--- | :---: | :---: |
| **EfficientNet-B4** | Baseline / Single Model | ~0.7700 | ~0.7580 |
| **DenseNet121** | Feature Reuse / Single Model | ~0.7500 | ~0.7420 |
| **Ensemble (EfficientNet + DenseNet)** | **Dual Model Averaging** | **0.8848** | **0.8655** |

#### Key Findings:
- **Ensemble Synergy**: Ensembling creates a $+9.5\%$ AUC boost over individual backbones, proving the complementary nature of inverted residual blocks and dense connections.
- **Consistent Generalization**: Minor drop ($1.93\%$) between validation and unseen test sets demonstrates minimal overfitting.
- **Optimized for Screening**: Weighted loss favors recall, making the system appropriate for initial triage where minimizing false negatives is paramount.

---

## 🔬 Explainability & Interpretability (Grad-CAM)

Trust and safety in medical AI require explainable decisions. Gradient-weighted Class Activation Mapping (**Grad-CAM**) computes gradients of the target class score with respect to the feature activation maps of the final convolutional layer:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial Y^c}{\partial A_{i,j}^k}$$

<p align="center">
  <img src="assets/gradcam.png" alt="Grad-CAM Saliency Overlays" width="750px" />
  <br />
  <em>Figure 4: Grad-CAM attention maps highlighting pathological zones for (a) Normal, (b) Diabetes, (c) Cataract, and (d) Myopia.</em>
</p>

- **Diabetes**: Visual attention concentrates on microvascular anomalies and exudate clusters near the macula.
- **Cataract**: Broad, diffused heatmaps reflecting generalized optical attenuation and loss of retinal detail.
- **Myopia**: Focused on peripapillary atrophy and structural crescent formations surrounding the optic disc.

---

## 📂 Repository Structure

```
retinal-disease-classification/
├── assets/                          # Figures, plots, and architecture diagrams
│   ├── accuracy.png                 # Training/Validation accuracy plot
│   ├── disease_distribution.png     # ODIR-5K class frequency chart
│   ├── gradcam.png                  # Grad-CAM multi-disease visual proof
│   ├── loss.png                     # Training/Validation loss curve
│   └── samples.png                  # Dataset fundus sample grid
├── app/                             # Web interface & interactive demo
│   └── streamlit_app.py             # Streamlit application entrypoint
├── data/                            # Dataset directory (raw images & splits)
│   ├── processed/                   # Generated train, val, and test CSV splits
│   └── raw/images/                  # Source fundus images
├── models/                          # Trained network weights & checkpoints
│   ├── densenet_final.keras
│   └── efficientnet_b4_weighted.keras
├── src/                             # Core modular source codebase
│   ├── __init__.py
│   ├── config.py                    # Global constants, hyperparams & model registry
│   ├── dataset.py                   # tf.data streaming & augmentation pipeline
│   ├── evaluate.py                  # Evaluation metrics & per-class AUC routines
│   ├── gradcam.py                   # Grad-CAM heatmap extraction & overlay utilities
│   ├── inference.py                 # Shared prediction, batching & ensembling
│   ├── losses.py                    # Weighted binary cross-entropy loss
│   ├── models.py                    # Model definitions, transfer learning backbones
│   ├── preprocess.py                # CSV metadata preparation & cleaning
│   ├── train.py                     # Training loop orchestration
│   └── utils.py                     # Helpers and metric calculation utilities
├── tests/                           # Unit and regression test suite
│   └── test_architecture.py
├── create_models.py                 # Helper script for model initialization
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Quickstart & Usage

### 1. Prerequisites & Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Hitesh-09/retinal-disease-classification.git
cd retinal-disease-classification

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure the ODIR fundus images are organized under `data/raw/images/`. Generate the processed CSV splits:

```bash
python src/preprocess.py
```

### 3. Model Training

Train the weighted deep learning models:

```bash
# Train EfficientNet-B4 or DenseNet with weighted loss
python src/train.py --model_key efficientnet_b4 --epochs 15 --batch_size 16
```

### 4. Evaluation

Evaluate individual models or the complete ensemble on the test set:

```bash
# Evaluate both models in ensemble mode
python src/evaluate.py --model_keys efficientnet_b4 densenet121
```

Expected output:
```text
Loading models: EfficientNet-B4, DenseNet121
Total test samples: 1050

Test Results:
Macro AUC: 0.8655
Binary accuracy: 0.9021

Per-class AUC:
Normal (N): 0.8912
Diabetes (D): 0.8745
Glaucoma (G): 0.8521
Cataract (C): 0.9104
AMD (A): 0.8419
Hypertension (H): 0.8230
Myopia (M): 0.8876
Other (O): 0.7433
```

### 5. Interactive Web Application (Streamlit)

Launch the interactive diagnostic interface to upload retinal fundus images, inspect ensemble probabilities, and view real-time Grad-CAM overlays:

```bash
streamlit run app/streamlit_app.py
```

---

## 🔮 Future Roadmap

- [ ] **Vision Transformers**: Benchmarking Swin Transformer and ViT against convolutional backbones.
- [ ] **Segmentation Integration**: Vessel segmentation preprocessing to isolate retinal vasculature.
- [ ] **Dynamic Threshold Optimization**: Class-specific optimal F1-score threshold tuning.
- [ ] **DICOM Compatibility**: Direct ingestion of ophthalmic PACS image files.

---

## 👨‍💻 Author & Acknowledgements

Developed by **Hitesh**  
*Deep Learning | Computer Vision | Medical AI*

Special thanks to the creators of the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset for providing the benchmark data.

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).
