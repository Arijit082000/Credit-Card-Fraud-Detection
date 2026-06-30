#  Credit Card Fraud Detection using Random Forest

## Project Overview
This project uses a **Random Forest Classifier** to identify fraudulent credit card transactions. The notebook covers data loading from OpenML, automated exploratory data analysis, preprocessing, model training, and evaluation with both the default prediction threshold and a custom threshold.

## Objective
- Detect fraudulent transactions accurately.
- Handle severe class imbalance using `class_weight='balanced'`.
- Compare model performance at different classification thresholds.

## Dataset
- **Source:** OpenML (`creditcard`, version 1)
- **Rows:** 284,807
- **Columns:** 30
- **Target:** `Class`
  - 0 = Legitimate
  - 1 = Fraud

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- YData Profiling

## Workflow
1. Load dataset from OpenML.
2. Inspect shape, info, missing values, and sample records.
3. Generate an automated EDA report using YData Profiling.
4. Split data into training and testing sets.
5. Train a Random Forest Classifier.
6. Evaluate using a confusion matrix and classification report.
7. Compare the default threshold (0.5) with a custom threshold (0.3).

## Model
- RandomForestClassifier
- `n_estimators=100`
- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1`

## Evaluation
- Confusion Matrix
- Classification Report
- Threshold comparison (0.5 vs 0.3)

## Project Structure
```
Random Forest on Credit Card Fraud Detection Data.ipynb
README.md
```

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling
```

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- SMOTE for class balancing
- XGBoost and LightGBM comparison
- ROC Curve and Precision-Recall Curve
- Feature importance visualization

## Author
**Arijit Dasgupta**
