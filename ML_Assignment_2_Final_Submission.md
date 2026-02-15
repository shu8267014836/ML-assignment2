# Machine Learning Assignment 2

Name: Shubham Agarwal  
Student ID: 2025aa05482

GitHub Repository Link:  
https://github.com/shu8267014836/ML-assignment2

Live Streamlit App Link:  

https://ml-assignment2-qeahpnmwufoqciwdtzgnfx.streamlit.app/


 1. Problem Statement

Implement multiple classification models on a chosen dataset and build an interactive Streamlit web application to demonstrate them. The application must: allow dataset upload (CSV) and use one classification dataset meeting the assignment constraints (minimum 12 features, minimum 500 instances); implement all six required models (Logistic Regression, Decision Tree Classifier, K-Nearest Neighbor Classifier, Naive Bayes Classifier, Random Forest, XGBoost); for each model compute Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC); provide a model selection dropdown, display evaluation metrics, and show a confusion matrix or classification report; and be deployable on Streamlit Community Cloud. The assignment was performed on BITS Virtual Lab; a screenshot of execution is included in the submission PDF.

Dataset (from README): UCI Adult (Census Income) – adult.data. Source: UCI Machine Learning Repository. Task: Binary classification – predict whether a person’s income is >50K or ≤50K. Instances: ~32,000+ rows. Features: 14. Target: Income class (<=50K or >50K, encoded 0/1). 

 2. Model Comparison Table (Test Data)

*From README – same train/test split and preprocessing. AUC = ROC-AUC; MCC = Matthews Correlation Coefficient.*

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|------------|--------|-----|-----|
| Logistic Regression | 0.8540 | 0.9511 | 0.8906 | 0.8540 | 0.8626 | — |
| Decision Tree | 0.8889 | 0.9446 | 0.8960 | 0.8889 | 0.8914 | — |
| K-Nearest Neighbors | 0.8661 | 0.9493 | 0.8920 | 0.8661 | 0.8728 | — |
| Naive Bayes | 0.7382 | 0.9339 | 0.8783 | 0.7382 | 0.7603 | — |
| Random Forest (Ensemble) | 0.9152 | 0.9684 | 0.9166 | 0.9152 | 0.9158 | — |
| XGBoost (Ensemble) | 0.9327 | 0.9761 | 0.9315 | 0.9327 | 0.9318 | — |

Best accuracy: XGBoost (0.9327).

---

 3. Model-wise Observations

*From README – observations on the performance of each model.*

- Logistic Regression: Good accuracy (0.85) and strong AUC (0.95). Benefits from feature scaling; fast to train. Performance is limited by linear decision boundary.
- Decision Tree: Strong accuracy (0.89) and AUC (0.94). Interpretable but can overfit; depth and pruning matter. Good balance of precision and recall.
- K-Nearest Neighbors: Solid accuracy (0.87) and AUC (0.95). Sensitive to scaling and choice of k; weighted/distance options help.
- Naive Bayes: Lower accuracy (0.74) but fast training. AUC (0.93); independence assumption may not hold well for all features.
- Random Forest (Ensemble): Strong accuracy (0.92) and best AUC (0.97) among non-XGBoost models. Robust to noise and missing values; feature importance is available.
- XGBoost (Ensemble): Best overall: highest accuracy (0.93), AUC (0.98), and F1 (0.93). Gradient boosting suits this dataset; tuning improves results further.

---

 4. Application Screenshots

[Insert your BITS Virtual Lab execution screenshot here.]

[Optional: Training performance / test performance with confusion matrix from your Streamlit app.]
