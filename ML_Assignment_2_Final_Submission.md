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
| Logistic Regression | 0.7310 | 0.8018 | 0.7963 | 0.7310 | 0.7481 | 0.4612 |
| Decision Tree | 0.8361 | 0.8780 | 0.8398 | 0.8361 | 0.8378 | 0.6712 |
| kNN | 0.8093 | 0.8761 | 0.8342 | 0.8093 | 0.8171 | 0.6184 |
| Naive Bayes | 0.7815 | 0.8378 | 0.8078 | 0.7815 | 0.7904 | 0.5628 |
| Random Forest (Ensemble) | 0.8487 | 0.9074 | 0.8535 | 0.8487 | 0.8507 | 0.6972 |
| XGBoost (Ensemble) | 0.8644 | 0.9219 | 0.8652 | 0.8644 | 0.8648 | 0.7286 |

Best accuracy: XGBoost (0.8644).

---

 3. Model-wise Observations

*From README – observations on the performance of each model.*

- Logistic Regression: Moderate accuracy (0.73); benefits from feature scaling. Fast to train. AUC (0.80) and MCC (0.46) indicate reasonable discrimination; performance is limited by linear decision boundary on this dataset.
- Decision Tree: Good accuracy (0.84) and AUC (0.88). Interpretable but can overfit; depth and pruning matter. MCC (0.67) shows a good balance of TP, TN, FP, FN.
- kNN: Solid accuracy (0.81) and AUC (0.88). Sensitive to scaling and choice of k; weighted/distance options help. MCC (0.62) is consistent with precision and recall.
- Naive Bayes: Decent accuracy (0.78) and fast training. AUC (0.84) and MCC (0.56) are lower than tree/ensemble models; independence assumption may not hold well for all features.
- Random Forest (Ensemble): Strong accuracy (0.85), best AUC (0.91) among non-XGBoost models, and high MCC (0.70). Robust to noise and missing values; feature importance is available.
- XGBoost (Ensemble): Best overall: highest accuracy (0.86), AUC (0.92), and MCC (0.73). Gradient boosting suits this dataset; tuning learning rate and depth improves results further.

---

 4. Application Screenshots

[Insert your BITS Virtual Lab execution screenshot here.]

[Optional: Training performance / test performance with confusion matrix from your Streamlit app.]
