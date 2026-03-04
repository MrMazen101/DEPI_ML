# 📄 Bank Customer Churn Prediction Pipeline

## Project Objective
Develop a production-ready machine learning pipeline to predict customer churn probability with high interpretability.

**Primary Metric:** ROC AUC (Continuous Probabilities)

---

## 🏗️ Pipeline Architecture Overview
This notebook simulates a real-world deployment scenario, adhering to anti-leakage principles and model transparency. The execution is divided into the following phases:

---

### 1. Environment & Setup

**Cell 1: Install & Imports**
Initializes the environment and installs high-performance gradient boosting libraries (XGBoost, LightGBM, CatBoost) and interpretability tools (SHAP, LIME). Global random seeds are locked to guarantee reproducibility.

---

### 2. Data Ingestion & EDA

**Cell 2: Load Data**
Ingests the raw dataset and immediately separates the target variable (`churn`) from features to prevent any accidental target leakage in subsequent steps.

**Cell 2.5: EDA Analysis**
Conducts statistical summaries and visualizations to understand class distribution (imbalance check), data types, and missing values.

---

### 3. Feature Engineering & Preprocessing

**Cell 3: Feature Engineering**
Creates new predictive signals derived from raw data (e.g., balance-to-salary ratios or product utilization scores).

**Cell 4: Preprocessing**
Implements automated `ColumnTransformers` to handle scaling for numerical features and one-hot encoding for categorical features within a secure pipeline.

---

### 4. Model Development

**Cell 4.5: Hyperparameter Tuning**
Uses search strategies (like Grid or Randomized Search) to find the optimal configuration for the models.

**Cell 5: Model Building**
Defines the base estimators, focusing on tree-based ensembles that typically handle tabular bank data effectively.

**Cell 6: Pipeline Construction**
Integrates preprocessing and the model into a single Scikit-Learn `Pipeline` object to ensure consistent transformation of future data.

**Cell 7: CV Training**
Executes Stratified K-Fold Cross-Validation to assess model stability and generalizability across different data folds.

**Cell 8: Final Training**
Performs the final fit on the full training dataset using the optimized parameters found during cross-validation.

---

### 5. Advanced Evaluation & Calibration

**Cell 8.5: Secondary Metrics**
Calculates F1-Score, Precision, and Recall to understand model performance beyond just AUC, focusing on the cost of false positives vs. false negatives.

**Cell 9: Threshold Analysis**
Determines the optimal probability threshold to convert continuous probabilities into binary "churn/no-churn" decisions based on business requirements.

**Cell 9.5: Probability Calibration**
Ensures that the predicted 0.7 probability actually corresponds to a 70% real-world likelihood of churn using calibration curves.

---

### 6. Model Interpretability (Explainable AI)

**Cell 10: Feature Importance**
Visualizes which features (e.g., Age, Number of Products) had the highest global impact on the model's decisions.

**Cell 10.5: Permutation Importance**
Measures feature importance by observing how shuffling a single feature affects the model's error rate.

**Cell 10.6: Partial Dependence**
Visualizes the marginal effect one or two features have on the predicted outcome.

**Cell 11: SHAP Analysis**
Uses Shapley values to provide a mathematically rigorous explanation of global feature contributions and individual predictions.

**Cell 11.5: LIME Analysis**
Provides local, human-interpretable explanations for specific customer predictions to show "why" a particular individual is flagged.

---

### 7. Business & Final Output

**Cell 12: Business Insights**
Translates technical findings into actionable strategies (e.g., "Customers with more than 3 products are 40% more likely to churn").

**Cell 13: Results Summary**
A final high-level dashboard of model performance and key findings.

**Cell 14: Final Submission**
Generates the final output file (e.g., `final_submission.csv`) containing customer IDs and their calculated churn probabilities for use by the bank's retention team.