# Customer Churn Prediction (E-Commerce / Telecom Dataset)

## 📌 Project Overview
This project implements a Machine Learning pipeline to predict customer churn using Logistic Regression.  
The goal is to identify customers likely to leave a subscription-based service.

---

## 📊 Dataset
The dataset used is the Telco Customer Churn dataset.

Target variable:
- `Churn` (Yes / No)

Features used:
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

---

## ⚙️ ML Pipeline

1. Data cleaning and preprocessing  
2. Handling missing values  
3. Feature conversion to numeric  
4. Train-test split (80/20)  
5. Logistic Regression model training  
6. Model evaluation  

---

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC Score

Example result:

Model Accuracy: ~76%

---

## 🚀 How to Run

1. Install dependencies:
# pip install pandas scikit-learn matplotlib
2. Run the training script:

---

## My Role

I independently implemented the full ML pipeline including data preprocessing, feature engineering, model training, evaluation, and debugging issues related to missing values and numeric conversion. The project was iteratively improved with scaling and advanced evaluation metrics.

---

## File Structure

- `train.py` → Core model implementation
- `telco_churn.csv` → Dataset file

---
