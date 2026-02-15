# ML Assignment 2 – Classification Models & Streamlit App

Repository: [https://github.com/2025aa05482-bits/ML-assignment2](https://github.com/2025aa05482-bits/ML-assignment2)

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

Dataset: UCI Adult (Census Income) – adult.data

- Source: UCI Machine Learning Repository (public).
- Task: Binary classification – predict whether a person’s income is >50K or ≤50K based on census-style attributes.
- Instances: ~32,000+ rows (meets minimum 500).
- Features: 14 (meets minimum 12). No header row in the raw file; the app assigns default column names when needed.
  - Numeric: age, fnlwgt (final weight), education-num, capital-gain, capital-loss, hours-per-week.
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- Target: Income class – <=50K or >50K (binary). Encoded as 0/1 in the app.
- Missing values: Some fields contain ? (e.g. workclass, occupation, native-country). The app supports filling (mean/mode) or dropping rows with missing values.
- Preprocessing in the app: Target column is set (last column or auto-detected). Categorical features are label-encoded or one-hot encoded (one-hot when unique values ≤ 10). Optional feature scaling (StandardScaler, RobustScaler, MinMaxScaler) and feature selection (SelectKBest) are available.

---

 c. Models used

# Comparison table (evaluation metrics for all 6 models)

The following metrics were calculated on the chosen dataset (adult.data, UCI Adult) using the same train/test split and preprocessing. AUC refers to ROC-AUC; MCC is Matthews Correlation Coefficient.

| ML Model Name              | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression        | 0.7310   | 0.8018| 0.7963    | 0.7310 | 0.7481| 0.4612|
| Decision Tree               | 0.8361   | 0.8780| 0.8398    | 0.8361 | 0.8378| 0.6712|
| kNN                         | 0.8093   | 0.8761| 0.8342    | 0.8093 | 0.8171| 0.6184|
| Naive Bayes                 | 0.7815   | 0.8378| 0.8078    | 0.7815 | 0.7904| 0.5628|
| Random Forest (Ensemble)     | 0.8487   | 0.9074| 0.8535    | 0.8487 | 0.8507| 0.6972|
| XGBoost (Ensemble)          | 0.8644   | 0.9219| 0.8652    | 0.8644 | 0.8648| 0.7286|

Best accuracy: XGBoost (0.8644).

![All models comparison](assets/results_comparison.png)

---

# Observations on the performance of each model

| ML Model Name               | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression     | Moderate accuracy (0.73); benefits from feature scaling. Fast to train. AUC (0.80) and MCC (0.46) indicate reasonable discrimination; performance is limited by linear decision boundary on this dataset. |
| Decision Tree           | Good accuracy (0.84) and AUC (0.88). Interpretable but can overfit; depth and pruning matter. MCC (0.67) shows a good balance of TP, TN, FP, FN. |
| kNN                     | Solid accuracy (0.81) and AUC (0.88). Sensitive to scaling and choice of k; weighted/distance options help. MCC (0.62) is consistent with precision and recall. |
| Naive Bayes             | Decent accuracy (0.78) and fast training. AUC (0.84) and MCC (0.56) are lower than tree/ensemble models; independence assumption may not hold well for all features. |
| Random Forest (Ensemble) | Strong accuracy (0.85), best AUC (0.91) among non-XGBoost models, and high MCC (0.70). Robust to noise and missing values; feature importance is available. |
| XGBoost (Ensemble)      | Best overall: highest accuracy (0.86), AUC (0.92), and MCC (0.73). Gradient boosting suits this dataset; tuning learning rate and depth improves results further. |

---

 Project structure


ML-assignment2/
├── app.py                 # Streamlit app (upload, model selection, metrics, confusion matrix)
├── requirements.txt       # Dependencies (streamlit, scikit-learn, pandas, numpy, xgboost, plotly, etc.)
├── README.md              # This file (Step 5 structure)
├── ML_Assignment_2.pdf     # Assignment document
├── adult.data             # UCI Adult dataset
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
   git clone https://github.com/2025aa05482-bits/ML-assignment2.git
   cd ML-assignment2
   

2. Install dependencies
   bash
   pip install -r requirements.txt
   

3. Run the Streamlit app
   bash
   streamlit run app.py
   

4. Open the URL shown (e.g. http://localhost:8501), upload adult.data (or a CSV version), set the target column (income/last column), choose a model from the dropdown, and view evaluation metrics and confusion matrix in the app.

---

*This README content is included in the submitted PDF as per Section 3 – Step 5 of the assignment.*
