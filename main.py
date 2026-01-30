"""
Earthquake Damage Prediction Pipeline
Predicts building damage grades (1-3) from structural features
"""

import pandas as pd
from pathlib import Path
from modelling import train_cv_model
from data_processing import (
    winsorize_dataframe,
    drop_columns,
    encode_labels,
    encode_features
)

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = Path("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data")
RESULTS_PATH = Path("./results.csv")
RANDOM_STATE = 42

# Columns that don't improve model performance or cause data leakage
COLUMNS_TO_DROP = [
    "count_floors_pre_eq",  # Redundant with other floor features
    "building_id",  # Not a predictive feature
    "plan_configuration",  # High cardinality, low signal
    "legal_ownership_status",  # Not predictive
    "has_secondary_use",  # Redundant parent feature
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other"
]

# ==============================================================================
# Load Data
# ==============================================================================
print("Loading data...")
train_X = pd.read_csv(DATA_DIR / "train_values.csv")
train_y = pd.read_csv(DATA_DIR / "train_labels.csv")
test_X = pd.read_csv(DATA_DIR / "test_values.csv")

print(f"Training set: {len(train_X)} samples")
print(f"Test set: {len(test_X)} samples")

# Convert labels from 1-3 to 0-2 for model compatibility
encoded_train_y = encode_labels(train_y["damage_grade"])


# ==============================================================================
# Preprocessing Pipeline
# ==============================================================================
def preprocess_dataset(df, y, df_test, columns_to_drop):
    """
    Preprocess training and test data with consistent transformations.

    Steps:
    1. Drop non-predictive or redundant columns
    2. Winsorize numerical features to handle outliers (clip at 5th/95th percentile)
    3. Encode categorical features (OneHot for low cardinality, Target for high cardinality)

    Args:
        df: Training feature DataFrame
        y: Encoded training labels (0-2)
        df_test: Test feature DataFrame
        columns_to_drop: List of column names to remove

    Returns:
        tuple: (processed_train_X, processed_test_X)
    """
    # Remove non-predictive features
    df = drop_columns(df, columns_to_drop)

    # Handle outliers in numerical features
    df = winsorize_dataframe(df)

    # Encode categorical features using training target
    # Note: encoder is fitted on training data only to prevent data leakage
    return encode_features(df, y, df_test)


print("\nPreprocessing data...")
train_X_processed, test_X_processed = preprocess_dataset(
    train_X,
    encoded_train_y,
    test_X,
    COLUMNS_TO_DROP
)

print(f"Features after preprocessing: {train_X_processed.shape[1]}")

# ==============================================================================
# Train Model with Cross-Validation
# ==============================================================================
print("\nTraining model with cross-validation...")
test_predictions = train_cv_model(
    train_X_processed,
    encoded_train_y,
    test_X_processed
)

# Convert predictions back to original scale (0-2 → 1-3)
test_predictions = test_predictions + 1

# ==============================================================================
# Save Results
# ==============================================================================
print("\nGenerating submission file...")
buildings_id = test_X["building_id"]

results = pd.DataFrame({
    "building_id": buildings_id,
    "damage_grade": test_predictions
})

results.to_csv(RESULTS_PATH, index=False)
print(f"Results saved to {RESULTS_PATH}")
print(f"Total predictions: {len(results)}")
print(f"Damage grade distribution:\n{results['damage_grade'].value_counts().sort_index()}")