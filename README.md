# 🌱 Plant Seedlings Classification Project

## Overview
This project focuses on **classifying plant seedlings** into their respective species using deep learning techniques. It compares the performance of three state-of-the-art convolutional neural network (CNN) architectures: **ResNet**, **Xception**, and **DenseNet**. 

The aim is to identify the best-performing model for accurate plant species classification while handling class imbalance and maximizing generalization.

---

## 📁 Dataset Details 
- **Classes**: 12 plant species.
- **Preprocessing**:
  - Resizing: All images resized to `224x224 pixels`.
  - Normalization: Pixel values scaled to `[0, 1]`.
  - Label Encoding: Plant species encoded and converted to categorical format.
  - Train-Test Split: Dataset split in an 80/20 ratio.
  - Class Weights: Computed to address dataset imbalance.
  - Data Augmentation: Applied techniques include rotation, zoom, flips, and shifts.
-For more information and to access the dataset, visit [Plant Seedlings Classification Dataset on Kaggle](https://www.kaggle.com/c/plant-seedlings-classification/data).
---

## 🛠️ Model Architectures

### 1️⃣ **ResNet (from Scratch)**
- **Core Features**:
  - Residual Blocks for mitigating vanishing gradient issues.
  - Custom top layers for task-specific feature learning.
  - Strong generalization aided by class weights and data augmentation.
- **Metrics**:
  - High performance on large datasets but requires careful regularization.

### 2️⃣ **Xception (Transfer Learning)**
- **Highlights**:
  - Pre-trained on ImageNet for feature extraction.
  - Lightweight and computationally efficient.
  - Custom dense layers added for the specific task.
- **Advantages**:
  - Faster convergence and reduced training time.

### 3️⃣ **DenseNet121 (Transfer Learning)**
- **Innovations**:
  - Dense Connectivity: Feature reuse across layers.
  - Pre-trained weights improve performance on limited datasets.
- **Benefits**:
  - Superior accuracy and AUC scores.
  - Fewer parameters compared to ResNet.
- **Trade-offs**:
  - Requires more memory and is computationally intensive.

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Confusion Matrix**
- **ROC & AUC Curves**
- **Precision, Recall, F1-score**
- **Training/Validation Loss & Accuracy Curves**

---

## 🔑 Key Findings
- **DenseNet121**: Achieved the best overall performance, making it the recommended model for final implementation.
- **Xception**: A strong contender, notable for its efficiency.
- **ResNet**: Performed well but lagged slightly due to training from scratch on limited data.

---

## ⚙️ Project Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hassona18/plant-seedlings-classification.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Code**:
   ```bash
   python plant-seedlings-classification.ipynb
   ```

---

## 🧑‍💻 Contributors
- **Mohab**
- **Sayed Saber**
- **Saeed Saber**
- **Hassan Anees**
- **Ashraf**
- **Khaled**

---

## 📭 Contact
- **Email**: [hassananees188@gmail.com](mailto:hassananees188@gmail.com)
- **Website**: [fgffs3444.com](https://www.fgffs3444.com)
- **Phone**: +20 1144438800

---

