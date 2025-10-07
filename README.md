# 🧠 Brain Tumor Detection using MRI Images

This project aims to **classify MRI brain images** as either containing a tumor or not using **deep learning (Transfer Learning with EfficientNetB0)**.  
It performs **binary classification** — predicting whether an MRI scan shows the presence of a brain tumor.

---

## 📁 Dataset

**Dataset:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
**Author:** Navoneel Chakrabarty  
**License:** CC0: Public Domain  
**Source:** Kaggle  

### Dataset Details
- **Classes:** `yes` (tumor) and `no` (no tumor)
- **Total Images:** ~3,000 MRI scans
- **Image Type:** Grayscale / RGB MRI images
- **Task Type:** Binary Classification
- **Structure:**




---

## ⚙️ Project Overview

| Step | Description |
|------|--------------|
| **1. Data Loading & Preprocessing** | Organized dataset into Pandas DataFrame with image paths and labels. |
| **2. Train-Validation Split** | Used 80% for training, 20% for validation. |
| **3. Data Augmentation** | Applied rotation, zoom, flip, and rescale using `ImageDataGenerator`. |
| **4. Transfer Learning Model** | Based on **EfficientNetB0** pre-trained on ImageNet. |
| **5. Fine-Tuning** | Unfroze top 50 layers for domain-specific learning. |
| **6. Evaluation** | Used validation accuracy, loss curves, confusion matrix, and Grad-CAM visualization. |

---

## 🧩 Model Architecture

- **Base Model:** `EfficientNetB0` (from `keras.applications`)
- **Top Layers:**
- Global Average Pooling
- Dense(256, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')
- **Optimizer:** Adam (learning rate = 1e-4 initially, 1e-5 during fine-tuning)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

---

## 🧠 Training Summary

| Phase | Epochs | Train Acc | Val Acc | Val Loss |
|-------|---------|------------|----------|-----------|
| Initial Training | 20 | ~0.91 | ~0.86 | 0.31 |
| Fine-tuning | 20 | ~0.95 | ~0.91 | 0.24 |

---

## 📈 Results & Visualization

### **1. Accuracy and Loss Curves**

These plots show training vs validation accuracy and loss over epochs.

![Accuracy and Loss Curves](results/accuracy_loss.png)

- Training accuracy gradually improved to **~94%**
- Validation accuracy stabilized around **~91%**
- Both training and validation loss decreased steadily

---

### **2. Confusion Matrix**

Displays the number of correctly and incorrectly classified MRI scans.

| Metric | Value |
|---------|--------|
| Precision | High (~0.90+) |
| Recall | High (~0.88+) |
| F1-score | ~0.89 |

Model correctly classifies most tumor and non-tumor images.

---

### **3. Grad-CAM Heatmaps**

Grad-CAM visualizations highlight **tumor regions** the model focuses on when making predictions.

#### 🔥 Original + Heatmap Overlay
| MRI Image | Grad-CAM Heatmap | Predicted Tumor Probability |
|------------|------------------|-----------------------------|
| Image 1 | Overlay Heatmap | 0.65 |
| Image 2 | Overlay Heatmap | 0.07 |
| Image 3 | Overlay Heatmap | 0.76 |
| Image 4 | Overlay Heatmap | 0.98 |

These show that the model accurately identifies tumor areas with **high confidence**.

---

## 🚀 How to Run the Project

You can run the entire notebook directly on **Kaggle**:

1. Open the Kaggle Notebook editor.
2. Upload this notebook.
3. Add the dataset:  
 > “Brain MRI Images for Brain Tumor Detection” by Navoneel Chakrabarty  
4. Run all cells sequentially.

---

## 🧪 Technologies Used

| Library | Purpose |
|----------|----------|
| TensorFlow / Keras | Model training & architecture |
| OpenCV | Image preprocessing |
| Matplotlib | Visualization (accuracy/loss, Grad-CAM) |
| Pandas & NumPy | Data handling |
| scikit-learn | Evaluation (confusion matrix, metrics) |

---

## 🧾 Observations

- The model performs **strongly** on the limited dataset.  
- Validation accuracy is slightly lower due to small validation sample size.
- Model generalizes well and highlights tumor regions effectively.
- Increasing dataset diversity and applying stronger regularization (dropout, early stopping) can further improve results.

---

## 🧠 Conclusion

This project demonstrates how **transfer learning** can be effectively used for **medical image classification** with limited data.  
The model achieves high accuracy and provides **interpretable Grad-CAM heatmaps** that align with tumor regions in MRI scans.

> ✅ **Final Model Accuracy:** ~91–92% (Validation)  
> ✅ **Task:** Binary Brain Tumor Classification  
> ✅ **Framework:** TensorFlow / Keras  

---


