# 📦 Late Delivery Risk Prediction – Supply Chain Machine Learning

## 🔗 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset Description](#-dataset-description)
- [Tech Stack](#-tech-stack)
- [Key Challenges Addressed](#-key-challenges-addressed)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Modeling Approach](#-modeling-approach)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Explainability & Feature Importance](#-explainability--feature-importance)
- [Production Readiness & Inference](#-production-readiness--inference)
- [Project Structure](#-project-structure)
- [Deployment Considerations](#-deployment-considerations)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 📌 Project Overview
Late deliveries are a major operational challenge in supply-chain management, impacting customer satisfaction, logistics cost, and service reliability.

This project implements an **end-to-end machine learning pipeline** to predict **late delivery risk at the time of order placement**, enabling proactive intervention before fulfillment begins.

The solution is designed with **production-level best practices**, including leakage prevention, class imbalance handling, robust validation, explainability, and batch inference.

---

## 🎯 Business Problem
**Objective:**  
Predict whether an order is likely to be delivered late **before shipment execution**.

This is an **order-level, short-term operational decision problem**, best addressed using **binary classification**, not time-series forecasting.

### Business Impact
- Early identification of high-risk orders  
- Improved shipment prioritization  
- Reduced operational bottlenecks  
- Better customer experience  

---

## 🗂️ Dataset Description
- **Source:** DataCo Smart Supply Chain Dataset (Kaggle)
- **Records:** ~180,000 order-level transactions
- **Target Variable:** `Late_delivery_risk`
  - `0` → On-time delivery
  - `1` → Late delivery

Additional access-log data was provided but intentionally excluded, as it is more suitable for **demand forecasting**, which addresses a different decision horizon.

---

## 🛠 Tech Stack

### Programming & Analysis
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn

### Machine Learning
- scikit-learn
- XGBoost
- category-encoders

### Modeling & Evaluation
- Logistic Regression
- Random Forest
- XGBoost
- ROC-AUC, Precision, Recall, F1-score
- Stratified Train/Test Split
- Cross-Validation

### Engineering & Production
- Modular preprocessing & encoding pipelines
- joblib for model persistence
- Batch inference via CLI

---

## ⚠️ Key Challenges Addressed
- Class imbalance between late and on-time deliveries
- Data leakage from post-delivery features
- High-cardinality categorical variables
- Production-safe inference without retraining

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted with a **business-first approach**, focusing on:
- Target imbalance analysis
- Data quality and sanity checks
- Leakage identification
- Key numerical and categorical drivers
- Temporal patterns (weekday vs weekend behavior)

EDA findings were validated against **model feature importance** for consistency.

---

## 🤖 Modeling Approach
Three models were trained and evaluated:

| Model | Purpose |
|-----|--------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear benchmark |
| XGBoost | Final production model |

- **Primary metric:** ROC-AUC (robust to class imbalance)
- **Threshold tuning** applied to improve recall for late deliveries

---

## 📊 Model Evaluation & Results
- Confusion matrices were generated to analyze false positives and false negatives.
- ROC curve comparison shows consistent improvement from baseline models to XGBoost.
- XGBoost achieved the best balance between performance and generalization.

---

## 🧠 Explainability & Feature Importance
XGBoost feature importance highlights key drivers of late delivery risk, including:
- Customer and order location attributes
- Shipping and transaction types
- Product and category identifiers
- Missing-value indicators

These insights align strongly with EDA findings, improving model trustworthiness.

---

## 🚀 Production Readiness & Inference
The pipeline supports **batch inference without retraining**.

### Batch Inference Example
```bash
python app/predict.py \
  --input data/02-preprocessed/X_preprocessed.csv \
  --output data/04-predictions/inference_predictions.csv
