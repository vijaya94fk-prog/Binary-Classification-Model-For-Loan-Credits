# Credit Rating – Binary Classification

## Problem Statement
Credit App is a binary machine learning classification project to predict a target class based on input features.  
The objective is to train and evaluate **five machine learning models** and compare them using standard evaluation metrics.

---

## Dataset Description
The dataset was downloaded from Kaggle (`credit_customers.csv`) and consists of:

- **1000 samples**
- **20 input features**
- **1 binary target variable**

The dataset was **stratified and split** into:
- `credit_customer_train.csv` – 500 samples used for training
- `credit_customer_test.csv` – used for post-deployment evaluation

The stratified split ensures that the class distribution of the target variable is preserved across both training and testing datasets.

### Features
- Mix of **numerical** and **categorical** variables describing customer characteristics
- **Target variable**:  
  - `Good` credit customer  
  - `Bad` credit customer

---

## Points to Note
- **Class Imbalance**:  
  Minority class lies between **20% and 40%**, indicating *moderate imbalance*.  
  Class weights were applied for certain models.
- No **missing** or **duplicate** data
- Categorical variables were **encoded**
- Numerical features were **standardized**

---

## Models Used & Evaluation Metrics

| Model                 | Accuracy | Precision | Recall | F1-Score | AUC     | MCC     |
|----------------------|----------|-----------|--------|----------|---------|---------|
| Logistic Regression  | 0.64     | 0.4375    | 0.70   | 0.5385   | 0.7395  | 0.2883  |
| Decision Tree        | 0.61     | 0.3902    | 0.5333 | 0.4507   | 0.6719  | 0.1642  |
| kNN                  | 0.70     | 0.5000    | 0.3667 | 0.4231   | 0.6357  | 0.2318  |
| Naive Bayes          | 0.62     | 0.4048    | 0.5667 | 0.4722   | 0.6767  | 0.1945  |
| Random Forest        | 0.68     | 0.4000    | 0.1333 | 0.2000   | 0.7324  | 0.0727  |
| XGBoost              | 0.69     | 0.4815    | 0.4333 | 0.4561   | 0.6867  | 0.2408  |

---

## Observations on Model Performance

### Logistic Regression
- Correctly predicts **64%** of total instances
- Significant misclassification observed
- Precision of **43.75%** indicates relatively high false positives
- Recall of **70%** shows strong ability to identify positive cases
- Balanced F1-score
- Decent AUC; MCC is moderate  
**Overall**: Moderate performance

---

### Decision Tree
- **61% accuracy** with notable misclassification
- Precision of **39%** → many false positives
- Recall of **53%**
- AUC of **0.67** indicates better-than-chance class separation
- MCC is very weak  
**Overall**: Mediocre performance

---

### k-Nearest Neighbors (kNN)
- Highest accuracy at **70%**
- Precision at **50%**: predicts positives correctly only half the time
- Recall of **37%** → many false negatives
- F1-score slightly lower than Decision Tree
- AUC lower than Decision Tree, but MCC is better  
**Overall**: Mediocre performance

---

### Naive Bayes
- Accuracy of **62%**, slightly better than Decision Tree
- Precision of **40%** with many false positives
- Recall of **56%**, better than most models except Logistic Regression
- Balanced F1-score
- AUC of **67%** and weak MCC  
**Overall**: Moderate but inconsistent

---

### Random Forest (Ensemble)
- Accuracy of **68%** and decent AUC
- Extremely low recall (**13%**), missing **87%** of actual positives
- Poor MCC due to class imbalance  
**Overall**: Not a reliable classifier for this dataset

---

### XGBoost (Ensemble)
- Accuracy of **69%**
- Precision of **48%** indicates relatively high false positives
- Recall of **43%**
- Balanced F1-score
- Decent AUC and moderate MCC  
**Overall**: Moderate performance

---

## Conclusion
Among all models, **Logistic Regression and XGBoost** offer the most balanced performance across metrics, while **kNN** achieves the highest accuracy. Class imbalance significantly impacts ensemble models, especially Random Forest.

