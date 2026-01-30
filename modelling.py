"""
Model training and prediction module using XGBoost
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold


def train_cv_model(X_train, y, X_test):
    """
    Train an XGBoost model with cross-validation and return test predictions.

    Args:
        X_train: Training features
        y: Training labels (damage grades 0, 1, 2)
        X_test: Test features

    Returns:
        Test predictions (0, 1, 2)
    """

    # Initialize XGBoost classifier for multi-class classification
    model = XGBClassifier(
        objective='multi:softprob',  # Multi-class with probabilities
        num_class=3,                 # 3 damage grades
        eval_metric='mlogloss',      # Multi-class log loss
        random_state=42              # For reproducibility
    )

    # Train the model on full training set
    model.fit(X_train, y)

    # Set up 2-fold cross-validation with shuffling
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    # Evaluate model performance using F1-macro score
    # F1-macro gives equal weight to all classes
    scores = cross_val_score(model, X_train, y, cv=kf, scoring='f1_macro')

    # Uncomment to see cross-validation results:
    # print(f"CV Scores: {scores}")
    # print(f"Average F1-Macro: {scores.mean():.3f}")
    # print(f"Standard Deviation: {scores.std():.3f}")

    # Uncomment to visualize the first tree:
    # plot_tree(model, num_trees=0)
    # plt.show()

    # Generate predictions on test set
    predictions_test = model.predict(X_test)

    # Uncomment to evaluate training performance:
    # predictions_train = model.predict(X_train)
    # F1_train = f1_score(y, predictions_train, average='micro')
    # print(f'The F1 at training is {F1_train}')

    return predictions_test