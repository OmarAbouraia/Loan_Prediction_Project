# 🏦 Loan Prediction Machine Learning Project  
*A complete end-to-end data mining pipeline for predicting loan approval decisions.*

---

## 📘 Project Overview  
This project aims to build a machine learning system that predicts whether a customer’s loan application will be approved.  
Using a structured banking dataset of 614 records, we developed a full **data preprocessing pipeline**, applied multiple **modeling strategies**, handled **class imbalance**, and deployed a **final predictive pipeline** ready for new customer input.

This work is documented in two phases:  
- **Phase 1 — Data Preprocessing** :contentReference[oaicite:2]{index=2}  
- **Phase 2 — Modeling & Evaluation** :contentReference[oaicite:3]{index=3}  

---

## 🧹 Phase 1: Data Preprocessing  
Comprehensive preprocessing was performed to ensure data quality and model readiness.

### 🔍 1. Data Exploration  
- 614 rows, 13 attributes  
- Mixed numerical & categorical features  
- Target variable: `Loan_Status`  
- Missing values detected in multiple columns (Gender, Married, LoanAmount, etc.)  

### 🧭 2. Handling Missing Values  
- Mode-based imputation for most categorical variables  
- Logical rule-based imputation for *Married* (Graduate → Yes / Non-Graduate → No)  
- RandomForestRegressor used to **predict missing LoanAmount** (advanced imputation approach)  
- All missing values fully resolved

### 🛠 3. Feature Engineering  
- Created `TotalIncome` = ApplicantIncome + CoapplicantIncome  
- Log-transform applied to **LoanAmount** and **TotalIncome**  
- Gentle outlier capping at 99.5th percentile  
- These transformations improved feature symmetry and reduced skewness

### 🔐 4. Encoding & Scaling  
- Label Encoding for ordinal features (Dependents, Education)  
- One-Hot Encoding for nominal features (Gender, Married, Property_Area, etc.)  
- StandardScaler applied to selected numerical features  
- Resulting dataset was clean, encoded, scaled, and ready for model training  

---

## 🤖 Phase 2: Modeling Approaches  
Multiple supervised learning models were tested:

### ✔ Models Tested
- **Decision Tree Classifier**  
- **K-Nearest Neighbors (KNN)**  
- **Naïve Bayes (Gaussian, Multinomial, Bernoulli)**  
- **XGBoost Classifier**

Each model was evaluated with various imbalance-handling strategies.

### ⚖ Class Imbalance Handling
Loan approvals (1) ≈ 2× loan rejections (0).  
To address this, we used:
- **SMOTE**
- **ADASYN**
- **SMOTE + Tomek Links**
- **Class Weights**
- **Decision Threshold Tuning**

### 📈 Notable Insights from Experiments
- Gaussian NB provides strong baseline performance but is biased toward majority class  
- Multinomial & Bernoulli NB are not suitable for continuous financial data  
- KNN improves significantly after scaling + SMOTE  
- XGBoost achieved high performance but lacked interpretability  
- Decision Tree with **balanced class weights** showed the best balance of:
  - Accuracy  
  - Interpretability  
  - Minority class detection  
  - Stability across strategies  

---

## 🏆 Final Selected Model  
### 🎯 **Decision Tree Classifier (with Balanced Class Weights)**  
This model was selected because:  
- It performs competitively across all evaluation metrics  
- It handles nonlinear financial relationships naturally  
- It is fully interpretable — essential for financial decision-making  
- It remains stable under class imbalance  
- It achieved strong validation results across customer segments

---

## 🔄 Deployment Pipeline  
A full prediction pipeline was implemented, including:

- Saved imputers (joblib)  
- Training column alignment  
- LoanAmount regression model  
- Encoders & scaler  
- Final Decision Tree model  

A new customer’s data passes through:
1. Preprocessing
2. Feature engineering
3. Encoding
4. Scaling
5. Final prediction

This makes the system ready for deployment or integration into a web or mobile app.

---

## 📊 Business Validation  
The model was tested on a real-world inspired subgroup:  
**Married + Semiurban applicants**.

- Historical approval rate ≈ 80%  
- Model predicted ≈ 75% approvals  
→ The model generalizes well and behaves realistically.

---

## 📁 Repository Structure
