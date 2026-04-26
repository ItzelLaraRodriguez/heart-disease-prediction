# Heart Disease Prediction (Machine Learning Project)

## Overview

This project builds a machine learning model to predict the likelihood of heart disease using anonymized patient data. The goal is to identify high-risk patients while prioritizing recall, ensuring that as few cases of heart disease as possible are missed.

---

## Dataset

* Source: UCI / Kaggle Heart Disease Dataset
* Records: 918 patients
* Target variable: `HeartDisease` (1 = disease, 0 = no disease)
* Features include age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more

---

## Exploratory Data Analysis

Initial analysis revealed several patterns:

* Exercise-induced angina is strongly associated with heart disease
* ST segment slope (Flat) shows a high correlation with heart disease
* Oldpeak (ST depression) is an important indicator of cardiac stress

---

## Data Cleaning

Data quality issues were identified and addressed:

* Records with `RestingBP = 0` were removed as physiologically invalid
* `Cholesterol = 0` values were replaced with the median value
  These decisions were supported by visual analysis of feature distributions

---

## Methodology

### Preprocessing

* One-hot encoding for categorical variables
* Feature scaling using `StandardScaler`

### Models

The following models were trained and evaluated:

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest

---

## Results

| Model               | Accuracy | Recall (Heart Disease) |
| ------------------- | -------- | ---------------------- |
| KNN (K=11)          | 0.87     | 0.87                   |
| Logistic Regression | 0.87     | 0.86                   |
| Random Forest       | 0.88     | 0.90                   |

---

## Model Selection

Random Forest was selected as the final model due to its higher recall. In this context, recall is prioritized because failing to identify a patient with heart disease (false negative) has more serious consequences than incorrectly flagging a healthy patient.

---

## Key Takeaways

* Model evaluation should reflect real-world priorities, not just accuracy
* Feature scaling significantly improves distance-based models such as KNN
* Ensemble methods such as Random Forest can capture more complex relationships

---

## Future Work

* Evaluate models using ROC-AUC
* Explore additional models such as gradient boosting
* Address potential class imbalance explicitly
* Develop a simple interface for predictions

---

## Technologies Used

* Python
* pandas, numpy
* scikit-learn
* matplotlib, seaborn

---

Itzel Lara Rodriguez
