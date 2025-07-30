# Loan Approval Prediction Model

## Project Overview

This project focuses on developing a machine learning model to **predict loan approval status** (`1` for Approved, `0` for Rejected) based on various applicant and loan-related features. The goal is to enhance and automate the loan assessment process, achieving a balance between profitability and risk management in financial institutions.

---

## Dataset

The dataset used is `loans_modified.csv`, containing:

- **563 entries**
- **13 features**

Key variables include:

- `applicant_income`, `coapplicant_income`, `loan_amount`
- `credit_history`, `married`, `education`, `property_area`

---

## Key Challenges & Insights from EDA

Exploratory Data Analysis (EDA) revealed several insights:

- **Missing Data**: Present in all columns — handled with targeted imputation.
- **Duplicate Records**: 25 duplicates identified and removed.
- **Class Imbalance**:  
  - **71.8% Approved**  
  - **28.2% Rejected**  
  Bias risk addressed during modeling.
- **Data Skewness & Outliers**:  
  `applicant_income`, `coapplicant_income`, and `loan_amount` showed strong right-skewness and outliers.
- **Strong Predictors**:
  - `credit_history`: **80% approval rate** with good credit.
  - Demographics: Married, male, and graduate applicants fared better.
  - Property Area: **Semiurban** properties saw the most approvals.

---

## Data Preprocessing Steps

A comprehensive pipeline was followed:

1. **Dropped Irrelevant Features**:
   - `loan_id`
2. **Data Cleaning**:
   - `dependents`: "3+" → `3.0`
3. **Handled Duplicates**:
   - Removed 25 duplicate rows
4. **Missing Value Imputation**:
   - Dropped rows with missing `loan_status`
   - Categorical: Imputed with **mode**
   - Numerical: Imputed with **median**
5. **Log Transformation**:
   - Applied `np.log1p()` to skewed features
6. **Encoding Categorical Features**:
   - Used `pd.get_dummies(..., drop_first=True)`
7. **Feature Engineering**:
   - `total_income` = `applicant_income + coapplicant_income`
   - `loan_amount_per_income` = `loan_amount / total_income`
8. **Feature Scaling**:
   - Standardized all numerical features using `StandardScaler`

---

## Model Training and Evaluation

Data split: **80% training / 20% test**, stratified on `loan_status`.

| Model                  | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Key Observation                                                                 |
|------------------------|----------|-----------|--------|----------|---------|----------------------------------------------------------------------------------|
| **Logistic Regression** | 0.814    | 0.796     | 0.988  | 0.882    | 0.730   | High recall for 'Approved'; may misclassify 'Rejected'.                         |
| **Random Forest**       | 0.8052   | 0.7901    | 0.9880 | 0.8790   | **0.7424** | Balanced model; uses `class_weight='balanced'`.                                |
| **XGBoost Classifier**  | 0.7876   | 0.7870    | 0.9524 | 0.8617   | **0.7674** | Best AUC-ROC; good for distinguishing approval status; used `scale_pos_weight`. |

### Additional Evaluation Notes:

- **XGBoost** achieved the **highest AUC-ROC**, indicating strongest classification boundary.
- **Feature Importance (XGBoost)**:
  - `loan_amount_per_income`
  - `applicant_income`
  - `loan_amount`

---

## Conclusion and Recommendations

The project successfully built multiple models. The **XGBoost Classifier** stood out due to its:

- Robust AUC-ROC
- Balanced performance
- High predictive power

### Recommendations:

1. **Deploy XGBoost**: Further fine-tune with hyperparameter optimization.
2. **Emphasize `credit_history`**: A consistent and vital predictor.
3. **Use Engineered Features**: Incorporate `total_income` and `loan_amount_per_income` into underwriting processes.
4. **Handle Imbalance Actively**: Consider SMOTE or other techniques to rebalance data if rejection accuracy becomes critical.
5. **Retrain Regularly**: Establish monitoring pipelines to maintain performance with evolving data.
6. **Model Explainability**:
   - Use SHAP/LIME for interpretability and regulatory transparency.

---

> **Note**: For detailed code, performance visualizations (like ROC curves), and SHAP plots, refer to the accompanying Jupyter Notebook in this repository.
