# 🧠 Retinal Disease Classification using Deep Learning (ODIR Dataset)

## 📌 Overview

This project focuses on **multi-label retinal disease classification** using fundus images from the **ODIR dataset**.
The goal is to detect multiple eye diseases simultaneously using deep learning models and improve performance through **class imbalance handling, model ensembling, and explainability techniques**.

---

## 🚀 Key Features

* Multi-label classification (multiple diseases per image)
* Deep CNN architectures:

  * EfficientNet-B4
  * DenseNet121
* Class imbalance handling using **weighted loss**
* Ensemble learning for improved performance
* Grad-CAM visualization for model explainability
* Full evaluation:

  * AUC score
  * Per-class metrics
  * Confusion matrices

---

## 🧩 Dataset

* Dataset: **ODIR (Ocular Disease Intelligent Recognition)**
* Contains retinal fundus images labeled with multiple diseases

### Classes:

| Label | Disease                          |
| ----- | -------------------------------- |
| N     | Normal                           |
| D     | Diabetes                         |
| G     | Glaucoma                         |
| C     | Cataract                         |
| A     | Age-related Macular Degeneration |
| H     | Hypertension                     |
| M     | Myopia                           |
| O     | Other diseases                   |

---

## 🏗️ Project Structure

```
retinal-disease-classification/
│
├── data/
│   ├── raw/images/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── efficientnet_b4_weighted.keras
│   ├── densenet_final.keras
│   └── ensemble_model.keras
│
├── notebooks/
│   └── training_and_evaluation.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess.py
│
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Preparation

* CSV-based dataset creation
* Train/Validation/Test split (70/15/15)
* Image preprocessing and resizing

---

### 2. Handling Class Imbalance

* Severe imbalance observed across disease classes
* Applied **weighted binary cross-entropy loss**
* Improved detection of rare diseases

---

### 3. Model Training

#### EfficientNet-B4

* Pretrained on ImageNet
* Fine-tuned on retinal dataset

#### DenseNet121

* Complementary architecture
* Trained using same pipeline

---

### 4. Ensemble Learning

* Combined predictions of both models
* Averaged outputs for final prediction

---

### 5. Explainability (Grad-CAM)

* Visualized regions influencing predictions
* Provided interpretability for medical validation

---

## 📊 Results

### Model Performance (AUC)

| Model                      | AUC        |
| -------------------------- | ---------- |
| EfficientNet-B4 (Baseline) | ~0.77      |
| DenseNet121                | ~0.75      |
| Ensemble (Validation)      | **0.8848** |
| Ensemble (Test)            | **0.8655** |

---

### Key Observations

* Ensemble significantly improves performance
* Class imbalance handling improves minority detection
* Slight drop from validation to test → good generalization
* Higher false positives indicate increased sensitivity

---

## 📈 Evaluation Metrics

* Macro AUC
* Precision, Recall, F1-score (per class)
* Confusion Matrix (per class)

---

## ⚠️ Discussion

* Model shows **high sensitivity**, suitable for screening tasks
* False positives are higher due to emphasis on recall
* Threshold tuning improves precision-recall balance

---

## 🔬 Grad-CAM Visualization

Grad-CAM highlights regions in retinal images where the model focuses while making predictions, providing interpretability.

---

## 🧪 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/Hitesh-09/retinal-disease-classification.git
cd retinal-disease-classification
```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python pandas numpy matplotlib scikit-learn
```

### 3. Run Training

```bash
python src/train.py
```

### 4. Evaluate Model

```bash
python src/evaluate.py
```

---

## 💡 Future Work

* Fine-tuning with advanced augmentation
* Adding attention mechanisms
* Using larger ensembles
* Clinical validation with expert annotations

---

## 🧾 Conclusion

This project demonstrates an effective deep learning pipeline for multi-label retinal disease classification.
The use of **class imbalance handling, ensemble learning, and explainability techniques** significantly improves model performance and interpretability, making it suitable for real-world medical screening applications.

---

## 👨‍💻 Author

**Hitesh**
Deep Learning Enthusiast | Computer Vision | Medical AI

---

## ⭐ If you found this useful

Give this repo a star ⭐ and share it!
