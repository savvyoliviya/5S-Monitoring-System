# Credit Card Fraud Detection

A machine learning notebook that classifies credit card transactions as legitimate or fraudulent, and compares four classification models on the same data.

Notebook: `Credit_Card_Fraud_Detection.ipynb` (built for Google Colab)

## Overview

Credit card fraud is rare but costly. This project trains and compares four classifiers on the well-known European credit card transactions dataset (`creditcard.csv`). The workflow covers data loading, cleaning, feature scaling, feature selection, model training, evaluation and a simple visual comparison.

## Dataset

- **File:** `creditcard.csv` (not included in this repo; download it from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Columns:** `Time`, `V1`–`V28` (anonymized PCA components), `Amount`, and the target `Class`
- **Target:** `Class` = 1 for fraud, 0 for legitimate
- **Class balance:** extremely imbalanced; fraud is a tiny fraction of all transactions

To keep Colab from running out of memory, the notebook randomly samples **20,000 rows** (`random_state=42`).

## Pipeline

1. **Load data** by uploading `creditcard.csv` through the Colab file picker.
2. **Sample** 20,000 rows.
3. **Clean:** check for nulls (none found), fill any missing values with column means, and label-encode any text columns.
4. **Feature engineering:** add `new_feature` (product of the first two columns, `Time` × `V1`).
5. **Scale** all features with `StandardScaler`.
6. **Select features:** keep the top 5 using `SelectKBest` with `f_classif`.
7. **Split** 80% train / 20% test (`random_state=42`).
8. **Train** four models:
   - Logistic Regression (`max_iter=500`)
   - Decision Tree (`max_depth=10`)
   - Random Forest (`n_estimators=50`, `max_depth=10`)
   - Support Vector Machine (linear kernel)
9. **Evaluate** with accuracy, precision, recall, F1 score (weighted) and a confusion matrix.
10. **Compare** models in a results table and a bar chart.
11. **PCA** (3 components) is applied to the selected features at the end as an extra step; it is not used for modeling.

## Results

Test set of 4,000 transactions (7 fraudulent):

| Model               | Accuracy | F1 (weighted) |
|---------------------|----------|---------------|
| Logistic Regression | 0.99900  | 0.99900       |
| Decision Tree       | 0.99900  | 0.99892       |
| Random Forest       | 0.99900  | 0.99900       |
| SVM (linear)        | 0.99925  | 0.99922       |

Fraud cases caught (true positives out of 7): Logistic Regression 5, Decision Tree 4, Random Forest 5, SVM 5.

## Important Notes and Limitations

- **Accuracy is misleading here.** With so few fraud cases, a model that predicts "legitimate" every time would score about 99.8%. All models look nearly identical on accuracy, and weighted averages are dominated by the majority class.
- **Small number of fraud samples in the test set (7)** means a single missed case swings results noticeably. Treat the model ranking as indicative, not conclusive.
- **Better metrics for this problem:** report precision, recall and F1 for the fraud class only (`average='binary'`), plus PR-AUC or ROC-AUC.
- **Possible improvements:**
  - Use the full dataset, or stratified sampling (`stratify=y`) so every split contains fraud cases
  - Handle imbalance with class weights, SMOTE or undersampling
  - Use cross-validation and hyperparameter tuning
  - Fit the scaler and feature selector on the training split only, to avoid data leakage
  - Reconsider `new_feature`; multiplying `Time` by `V1` has no clear meaning

## Requirements

- Python 3
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install locally with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

**On Google Colab**
1. Open the notebook in Colab.
2. Run the first cell and upload `creditcard.csv` when prompted.
3. Run the remaining cells in order.

**Locally**
1. Place `creditcard.csv` in the same folder as the notebook.
2. Remove or skip the first cell (`google.colab` upload).
3. Run the notebook in Jupyter.

## Project Structure

```
.
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.csv        # download separately
└── README.md
```

