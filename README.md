# 🏦 AML Transaction Detection

Detecting money laundering transactions using Machine Learning on banking transaction data.

## 📌 Problem

Binary classification: determine whether each transaction is **legitimate** (0) or **money laundering** (1) — based on historical transaction data in chronological order, with no data leakage. Severe Class imbalance (class 1: ~0.01%)

## 📊 Results

| Model | ROC AUC (Val) | ROC AUC (Test) |
|---|---|---|
| XGBoost | ~0.99+ | ~0.99+ |
| Decision Tree | 0.9933 | 0.9952 |
| Logistic Regression | 0.8398 | 0.8452 |

## ⚙️ Feature Engineering

- **Temporal:** `hour_sin/cos`, `day_sin/cos` — cyclic time encoding
- **Transaction:** `log_amount`, `is_cross_currency`, `payment_type_enc`
- **Historical (no leakage):** cumulative stats per sender/receiver — avg amount, std amount, unique counterparties, interaction count

## 🗂️ Data Split

Split in chronological order: **Train 60% / Val 20% / Test 20%**  
`PredefinedSplit` used for proper hyperparameter tuning.

## Feature Importance Analysis 
Use **XGB** and **SHAP**
