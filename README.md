# Mobile Game Player Spending Segment Classification (ANN & Traditional ML)

## Overview
This repository contains a Jupyter Notebook (`lab 12.ipynb`) that implements a multi-class classification pipeline to categorize mobile game players into spending segments (`Dolphin`, `Minnow`, `Whale`). The project trains and compares a custom **Artificial Neural Network (ANN)** against four traditional machine learning classifiers.

## Dataset
The models are trained on the `mobile_game_inapp_purchases.csv` dataset, which consists of **3,024 user records** across **13 original features**:
* **Demographics & Device:** `UserID`, `Age`, `Gender`, `Country`, `Device`
* **Gameplay Metrics:** `GameGenre`, `SessionCount`, `AverageSessionLength`
* **Monetization & Purchase History:** `SpendingSegment` (Target), `InAppPurchaseAmount`, `FirstPurchaseDaysAfterInstall`, `PaymentMethod`, `LastPurchaseDate`

## Pipeline & Methodology
1. **Data Cleaning & Imputation:**
   * Dropped non-predictive identifier and date columns (`UserID`, `LastPurchaseDate`).
   * Imputed missing numerical values (`Age`, `InAppPurchaseAmount`, `FirstPurchaseDaysAfterInstall`) using the median.
   * Imputed missing categorical values (`Gender`, `Country`, `Device`, `GameGenre`, `PaymentMethod`) using the mode.
2. **Feature Engineering & Preprocessing:**
   * Label-encoded the target variable `SpendingSegment` into 3 classes (`0: Dolphin`, `1: Minnow`, `2: Whale`).
   * Applied one-hot encoding (`OneHotEncoder`) to categorical features, expanding the feature space to **59 features**.
   * Standardized all numerical features using `StandardScaler`.
   * Split the dataset into training (80% — 2,419 samples) and testing (20% — 605 samples) sets using stratified sampling (`random_state=42`).
3. **Model Architectures:**
   * **Artificial Neural Network (ANN):** A Sequential TensorFlow/Keras model with an explicit input layer (59 features), three hidden `Dense` layers (`128 -> 64 -> 32` units with `ReLU` activation), and a 3-unit `Softmax` output layer (18,115 total trainable parameters). Trained for 50 epochs (`batch_size=32`, `validation_split=0.2`) using the Adam optimizer and sparse categorical cross-entropy loss.
   * **Traditional ML Baselines:**
     * Logistic Regression (`max_iter=1000`)
     * K-Nearest Neighbors (KNN)
     * Random Forest Classifier
     * Decision Tree Classifier
4. **Evaluation & Visualization:**
   * Plotted training vs. validation loss and accuracy learning curves for the ANN.
   * Evaluated all 5 models on the test set using confusion matrices, classification reports (Precision, Recall, F1-Score), and multi-class ROC-AUC curves.

## Tech Stack & Dependencies
To run this notebook locally or in Google Colab, ensure you have the following Python libraries installed:
* `pandas`
* `numpy`
* `scikit-learn`
* `tensorflow` / `keras`
* `matplotlib`
* `seaborn`

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
