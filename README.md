# 🚚 Supply Chain Late Delivery Risk Prediction (Machine Learning)

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset Description](#dataset-description)
- [Tech Stack](#tech-stack)
- [Key Challenges Addressed](#key-challenges-addressed)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
- [Encoding Strategy](#encoding-strategy)
- [Models & Evaluation](#models--evaluation)
- [Overfitting & Validation](#overfitting--validation)
- [Production Readiness & Inference](#production-readiness--inference)
- [Project Structure](#project-structure)
- [Deployment Considerations](#deployment-considerations)
- [Conclusion](#conclusion)
- [Author](#author)

---

## Project Overview
Late deliveries are a recurring operational challenge in large-scale supply chain systems. They negatively impact customer satisfaction, increase logistics costs, and create downstream service-level agreement (SLA) risks.

This project implements a **production-oriented machine learning pipeline** to predict whether an order is likely to be delivered late **before fulfillment begins**.  
The emphasis is on **data integrity, leakage prevention, robust validation, and inference readiness**, rather than purely optimizing accuracy.

---

## Business Problem
**Objective:**  
Predict the probability that an order will be delivered late using only information available at order creation time.

**Why this matters:**
- Enables early identification of high-risk orders
- Supports proactive shipment prioritization
- Improves logistics planning and operational efficiency
- Reduces downstream escalations and penalties

This is an **order-level operational decision problem**, modeled as a **binary classification task**.

---

## Dataset Description
- **Source:** DataCo Smart Supply Chain Dataset
- **Volume:** ~180,000 order records
- **Target Variable:** `Late_delivery_risk`
  - `0` → On-time delivery
  - `1` → Late delivery
- **Feature Categories:**
  - Customer & geography
  - Order & product attributes
  - Market & logistics information

Additional metadata and access log files are included for completeness and future extensibility.

---

## Tech Stack
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Encoding:** Category Encoders
- **Model Persistence:** Joblib
- **Environment Management:** Virtualenv
- **Version Control:** Git, GitHub

---

## Key Challenges Addressed
This project explicitly handles real-world machine learning challenges:

- **Data Leakage Prevention**  
  All post-delivery and post-shipment features were removed to ensure realistic predictions.

- **Class Imbalance**  
  Late deliveries are not evenly distributed. Class weighting and probability threshold tuning were applied.

- **High Cardinality Categorical Features**  
  Many categorical fields (e.g., city, product, region) required scalable encoding strategies.

- **Overfitting Control**  
  Cross-validation, regularization, and model complexity constraints were used to ensure generalization.

---

## Exploratory Data Analysis (EDA)
EDA was conducted to:
- Understand the distribution of late vs on-time deliveries
- Identify key features correlated with delivery delays
- Analyze missing data patterns and data quality issues

EDA insights directly influenced:
- Feature selection
- Encoding strategy
- Model choice

📓 Notebook: `notebooks/EDA.ipynb`

---

## Data Preprocessing & Feature Engineering
Preprocessing was designed to be **training-safe and inference-consistent**:

- Removal of leakage-prone columns
- Date-based feature extraction (day of week, month, weekend flag)
- Missing value handling:
  - Numerical features → median imputation
  - Categorical features → `"Unknown"`
- Missing-value indicator flags added where applicable

Processed datasets are stored separately to maintain pipeline clarity.

---

## Encoding Strategy
A hybrid encoding approach was used:

- **Low-cardinality categorical features:** One-Hot Encoding  
- **High-cardinality categorical features:** Target Encoding  
- **Numerical features:** Standard Scaling  

Encoding is fitted **only on training data** to prevent target leakage.

---

## Models & Evaluation
The following models were trained and evaluated:

| Model | Test ROC-AUC |
|------|-------------|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.84 |
| XGBoost | ~0.82 |

**Primary metric:** ROC-AUC  
Additional evaluation includes:
- Precision, Recall, F1-score
- Confusion matrices
- ROC curve comparison

Logistic Regression provides interpretability, while XGBoost captures complex non-linear patterns.

---

## Overfitting & Validation
Model robustness was evaluated using:
- Stratified train-test split
- 5-fold cross-validation
- Regularization and depth constraints
- Threshold tuning for imbalanced data

**Observation:**  
Random Forest showed mild overfitting, while Logistic Regression and XGBoost demonstrated more stable generalization.

---

## Production Readiness & Inference
The project includes a standalone inference pipeline:

- Trained models and encoders are serialized (`.pkl`)
- Batch predictions supported via command line
- Outputs saved as CSV for downstream systems

**Example:**
python app/predict.py \
  --input data/02-preprocessed/X_preprocessed.csv \
  --output data/04-predictions/inference_predictions.csv
  
## Project Structure

<img width="839" height="1405" alt="image" src="https://github.com/user-attachments/assets/131af617-97a0-4ca1-a9b7-41336df7c75b" />

## Conclusion
This project demonstrates an end-to-end machine learning system focused on realistic constraints, data correctness, and production usability.  
The solution is designed to align closely with enterprise-level supply chain analytics use cases, emphasizing robust modeling, leakage prevention, and deployable inference workflows rather than experimental results alone.

## Author
**Ankita Sarkar**  
Data & Machine Learning Engineer

