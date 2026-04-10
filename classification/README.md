
## What this script does
This script demonstrates a full binary classification evaluation workflow:
1. Generate synthetic classification data
2. Split data into training and test sets
3. Train a logistic regression model
4. Predict probabilities and class labels
5. Compute ROC curve and AUC
6. Compute confusion matrix
7. Plot ROC curve and confusion matrix

## Key concepts
- `y_test`: true labels in the test set
- `y_scores`: predicted probabilities for class 1
- `y_pred`: final predicted class labels
- `ROC`: shows the trade-off between TPR and FPR across thresholds
- `AUC`: measures the overall discrimination ability of the classifier
- `Confusion Matrix`: compares true labels with predicted labels

## Why this matters
This file helps me understand:
- how classification models are evaluated
- why probability scores are different from final predicted labels
- how ROC and AUC are connected to model discrimination
- how sensitivity and specificity are derived

## My learning summary
- ROC is not a prediction model itself; it is an evaluation tool
- AUC closer to 1 means stronger discrimination
- `predict_proba()` returns probabilities
- `predict()` returns final class labels
- confusion matrix is the basis for sensitivity, specificity, and accuracy