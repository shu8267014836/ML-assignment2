# ML Assignment 2 – Classification Models & Streamlit App

Repository: https://github.com/shu8267014836/ML-assignment2

This README follows the structure required in Section 3 – Step 5 of the assignment. The full assignment document is in ML_Assignment_2.pdf.

---

a. Problem statement

Implement multiple classification models on a chosen dataset and build an interactive Streamlit web application to demonstrate them. The application must:

- Allow dataset upload (CSV) and use one classification dataset meeting the assignment constraints (minimum 12 features, minimum 500 instances).
- Implement all six required models: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbor Classifier, Naive Bayes Classifier (Gaussian or Multinomial), Random Forest (Ensemble), and XGBoost (Ensemble).
- For each model, compute: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
- Provide a model selection dropdown, display evaluation metrics, and show a confusion matrix or classification report.
- Be deployable on Streamlit Community Cloud.

The assignment was performed on BITS Virtual Lab; a screenshot of execution is included in the submission PDF.

---

b. Dataset description

Dataset: adult.data (loan/credit classification)

- Source: Public dataset (included in repo as adult.data).
- Task: Binary classification – predict the target class (0/1) based on person and loan attributes.
- Instances: ~45,000+ rows (meets minimum 500).
- Features: 14 columns total; 13 features (meets minimum 12). The file has a header row and is comma-separated.
  - Numeric: person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score.
  - Categorical: person_gender, person_education, person_home_ownership, loan_intent, previous_loan_defaults_on_file.
- Target: Last column `Target` – binary (0/1). The app auto-detects it by name.
- Preprocessing in the app: Target column is set (auto-detected by name or last column). Categorical features are label-encoded or one-hot encoded (one-hot when unique values ≤ 10). Optional feature scaling (StandardScaler, RobustScaler, MinMaxScaler) and feature selection (SelectKBest) are available. Missing values can be filled (mean/mode) or rows dropped.

---

c. Models used

# Comparison table (evaluation metrics for all 6 models)

The following metrics were calculated on the chosen dataset (adult.data) using the same train/test split and preprocessing. AUC refers to ROC-AUC; MCC is Matthews Correlation Coefficient.

| ML Model Name              | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression        | 0.8540   | 0.9511| 0.8906    | 0.8540 | 0.8626| —     |
| Decision Tree              | 0.8889   | 0.9446| 0.8960    | 0.8889 | 0.8914| —     |
| K-Nearest Neighbors        | 0.8661   | 0.9493| 0.8920    | 0.8661 | 0.8728| —     |
| Naive Bayes                | 0.7382   | 0.9339| 0.8783    | 0.7382 | 0.7603| —     |
| Random Forest (Ensemble)   | 0.9152   | 0.9684| 0.9166    | 0.9152 | 0.9158| —     |
| XGBoost (Ensemble)         | 0.9327   | 0.9761| 0.9315    | 0.9327 | 0.9318| —     |

Best accuracy: XGBoost (0.9327)
 

# Observations on the performance of each model

| ML Model Name               | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Good accuracy (0.85) and strong AUC (0.95). Benefits from feature scaling; fast to train. Performance is limited by linear decision boundary. |
| Decision Tree               | Strong accuracy (0.89) and AUC (0.94). Interpretable but can overfit; depth and pruning matter. Good balance of precision and recall. |
| K-Nearest Neighbors         | Solid accuracy (0.87) and AUC (0.95). Sensitive to scaling and choice of k; weighted/distance options help. |
| Naive Bayes                 | Lower accuracy (0.74) but fast training. AUC (0.93); independence assumption may not hold well for all features. |
| Random Forest (Ensemble)    | Strong accuracy (0.92) and best AUC (0.97) among non-XGBoost models. Robust to noise and missing values; feature importance is available. |
| XGBoost (Ensemble)          | Best overall: highest accuracy (0.93), AUC (0.98), and F1 (0.93). Gradient boosting suits this dataset; tuning improves results further. |

---

Project structure


ML-assignment2/
├── app.py                 # Streamlit app (upload, model selection, metrics, confusion matrix)
├── requirements.txt       # Dependencies (streamlit, scikit-learn, pandas, numpy, xgboost, plotly, etc.)
├── README.md              # This file (Step 5 structure)
├── ML_Assignment_2.pdf     # Assignment document
├── adult.data             # Dataset (13 features, binary target; ~45K rows)
├── assets/                # Screenshots (e.g. results_comparison.png)
└── model/                 # Implementations of all 6 classifiers
    ├── logistic.py        # Logistic Regression
    ├── dt.py              # Decision Tree
    ├── knn.py             # K-Nearest Neighbors
    ├── nb.py              # Naive Bayes
    ├── rf.py              # Random Forest
    └── xgb.py             # XGBoost


---

How to run the project

1. Clone the repository
   bash
   git clone https://github.com/shu8267014836/ML-assignment2.git
   cd ML-assignment2
   

2. Install dependencies
   bash
   pip install -r requirements.txt
   

3. Run the Streamlit app
   bash
   streamlit run app.py
   

4. Open the URL shown (e.g. http://localhost:8501), upload adult.data (or a CSV with header), set the target column (Target/last column), choose a model from the dropdown, and view evaluation metrics and confusion matrix in the app.

---
 