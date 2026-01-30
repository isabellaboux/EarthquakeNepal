"""
Model Training Module for Earthquake Damage Prediction

This module handles model training, cross-validation, and prediction
using XGBoost for multi-class classification of building damage grades.

Author: Tony Carricarte
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


# ==============================================================================
# Model Training and Evaluation
# ==============================================================================

def train_cv_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = True
) -> np.ndarray:
    """
    Train XGBoost classifier with cross-validation and generate test predictions.

    Pipeline:
    1. Define XGBoost multi-class classifier
    2. Perform stratified k-fold cross-validation to assess generalization
    3. Train final model on full training set
    4. Generate predictions on test set

    Args:
        X_train: Training features (encoded, shape: [n_samples, n_features])
        y_train: Training labels (0, 1, 2 for damage grades)
        X_test: Test features (encoded, shape: [n_test_samples, n_features])
        n_splits: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        verbose: Whether to print CV scores and metrics (default: True)

    Returns:
        Test set predictions as numpy array (values: 0, 1, 2)

    Example:
        >>> predictions = train_cv_model(X_train, y_train, X_test)
        >>> # CV F1-Macro: 0.756 (+/- 0.012)
    """

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------

    # XGBoost multi-class classifier
    # - multi:softprob returns class probabilities for calibrated predictions
    # - mlogloss (cross-entropy) is standard for multi-class problems
    model = XGBClassifier(
        objective='multi:softprob',  # Multi-class classification with probabilities
        num_class=3,  # Three damage grades (0, 1, 2)
        eval_metric='mlogloss',  # Multi-class log loss
        max_depth=6,  # Maximum tree depth (controls complexity)
        learning_rate=0.1,  # Step size shrinkage (eta)
        n_estimators=100,  # Number of boosting rounds
        subsample=0.8,  # Subsample ratio of training instances
        colsample_bytree=0.8,  # Subsample ratio of features per tree
        random_state=random_state,  # Reproducibility
        n_jobs=-1  # Use all CPU cores
    )

    # -------------------------------------------------------------------------
    # Cross-Validation
    # -------------------------------------------------------------------------

    # Stratified K-Fold maintains class distribution across folds
    # This is crucial for imbalanced multi-class problems
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Evaluate model using F1-macro (equal weight to all classes)
    # F1-macro is better than accuracy for imbalanced classes
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1
    )

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"Metric: F1-Macro (equal weight to all damage grades)")
        print(f"Number of folds: {n_splits}")
        print(f"\nFold scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"Std Deviation: {cv_scores.std():.4f}")
        print(f"95% CI: [{cv_scores.mean() - 2 * cv_scores.std():.4f}, "
              f"{cv_scores.mean() + 2 * cv_scores.std():.4f}]")
        print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Train Final Model
    # -------------------------------------------------------------------------

    # Train on full training set for final predictions
    # Note: In production, consider using early stopping with validation set
    model.fit(X_train, y_train)

    if verbose:
        # Show training performance (sanity check for overfitting)
        train_predictions = model.predict(X_train)
        train_f1 = f1_score(y_train, train_predictions, average='macro')
        print(f"Training F1-Macro: {train_f1:.4f}")
        print("\nTraining Set Classification Report:")
        print(classification_report(
            y_train,
            train_predictions,
            target_names=['Grade 1 (Low)', 'Grade 2 (Medium)', 'Grade 3 (High)']
        ))

    # -------------------------------------------------------------------------
    # Generate Test Predictions
    # -------------------------------------------------------------------------

    test_predictions = model.predict(X_test)

    if verbose:
        print("\nTest Set Prediction Distribution:")
        unique, counts = np.unique(test_predictions, return_counts=True)
        for grade, count in zip(unique, counts):
            percentage = (count / len(test_predictions)) * 100
            print(f"  Grade {grade + 1}: {count:5d} ({percentage:5.2f}%)")
        print()

    return test_predictions


# ==============================================================================
# Alternative Training Functions
# ==============================================================================

def train_model_with_class_weights(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        verbose: bool = True
) -> np.ndarray:
    """
    Train XGBoost with sample weights to handle class imbalance.

    Use this function if class imbalance is significant (e.g., 60:30:10 ratio).
    Sample weights give more importance to minority classes during training.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        verbose: Print training information

    Returns:
        Test predictions
    """
    # Calculate sample weights (inverse of class frequency)
    sample_weights = compute_sample_weight('balanced', y_train)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )

    # Fit with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)

    if verbose:
        train_f1 = f1_score(
            y_train,
            model.predict(X_train),
            average='macro'
        )
        print(f"Training F1-Macro (with class weights): {train_f1:.4f}")

    return model.predict(X_test)


# ==============================================================================
# Utility Functions
# ==============================================================================

def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: list = None
) -> None:
    """
    Print comprehensive evaluation metrics for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional class names for report
    """
    if label_names is None:
        label_names = ['Grade 1 (Low)', 'Grade 2 (Medium)', 'Grade 3 (High)']

    print("\nClassification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Calculate different F1 averaging methods
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"F1-Macro (unweighted mean):    {f1_macro:.4f}")
    print(f"F1-Micro (overall accuracy):   {f1_micro:.4f}")