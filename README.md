# 🧠 EEG Sleep Stage Classification (Ultimate Expansion)

## 📘 Project Overview

This project implements an **exhaustive, state-of-the-art machine learning pipeline** to classify **five sleep stages** from short **178-sample EEG segments**. The goal is to push classification performance to the highest possible level using a rich combination of domain knowledge, statistical learning, deep learning, and interpretability techniques.

---

## 👥 Group Information

**Students:**
- Alouit Mohsine Abdelhakim  
- Kherraf Zyad  
- Bedoui Alaaeddine  
- Fadene Akram  
- Sayad Ahmed  
- Dib Abdelmounaim  

---

## 🧪 Pipeline Summary

### 📊 1. Exploratory Data Analysis (EDA)
- Histograms, boxplots, violin plots
- Spectrograms and STFT visualizations
- Correlation matrices and EEG channel networks

### 🔍 2. Feature Engineering
**Time-Domain Features**
- Mean, variance, skewness, kurtosis
- RMS, peak-to-peak amplitude, zero-crossing rate

**Frequency-Domain Features**
- FFT bandpowers (delta, theta, alpha, beta, gamma)

**Time-Frequency Features**
- Short-Time Fourier Transform (STFT)
- Continuous & Discrete Wavelet Transforms (CWT, DWT)

**Nonlinear Dynamics**
- Entropy measures (approximate, sample, permutation)
- Hurst exponent
- Higuchi fractal dimension
- Hjorth mobility and complexity

**Engineered Features**
- Composite statistical features
- Signal decomposition metrics

---

### 🧬 3. Data Augmentation
- Gaussian noise injection
- Time-shifting
- Amplitude scaling
- Window slicing

---

### 🧠 4. Dimensionality Reduction & Clustering
- PCA, t-SNE, UMAP
- KMeans clustering (used as meta-features)

---

### 📉 5. Feature Selection
- Recursive Feature Elimination (RFE)
- SelectKBest (based on mutual info, ANOVA)
- LASSO regularization

---

### 🤖 6. Models Used
- Classical ML:  
  `Random Forest`, `XGBoost`, `LightGBM`, `SVM`, `KNN`, `ExtraTrees`

- Deep Learning:  
  `1D-CNN`, `CNN + BiLSTM`, `Autoencoder + Classifier`

---

### 🧪 7. Hyperparameter Optimization
- Grid Search
- Randomized Search
- Bayesian Optimization (optuna / skopt)

---

### 🧬 8. Model Ensembling & Meta-Learning
- Voting Classifier (soft & hard)
- Blending
- Stacking with meta-learners

---

### 📈 9. Model Evaluation
- Stratified Nested Cross-Validation
- Confusion Matrices & ROC-AUC
- Learning curves and training dynamics

---

### 📏 10. Calibration & Uncertainty Estimation
- `CalibratedClassifierCV`
- `Monte Carlo Dropout` for Bayesian uncertainty

---

### 🧠 11. Interpretability
- SHAP (SHapley Additive Explanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- PDP (Partial Dependence Plots)
- ALE (Accumulated Local Effects)

---

### 🗃️ 12. Reporting & Output
- Predictions exported to CSV
- Model artifacts saved (joblib/h5/onnx)
- Full performance reports and metrics

---

## 🛠️ Tools & Technologies

- **Python**, **NumPy**, **Pandas**, **Scikit-learn**, **MNE**
- **Matplotlib**, **Seaborn**, **Plotly**
- **TensorFlow** / **Keras**, **PyTorch**
- **XGBoost**, **LightGBM**, **SHAP**, **LIME**, **Optuna**

---
