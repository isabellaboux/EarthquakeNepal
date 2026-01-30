"""
Data Processing Module for Earthquake Damage Prediction

This module provides preprocessing functions for the Nepal Earthquake dataset including:
- Outlier handling via winsorization
- Feature encoding (OneHot and Target encoding)
- Column dropping utilities
- Label encoding

Author: Tony Carricarte
"""

from typing import Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==============================================================================
# Outlier Handling
# ==============================================================================

def winsorize_dataframe(
        dataframe: pd.DataFrame,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95
) -> pd.DataFrame:
    """
    Apply winsorization to numerical columns to handle outliers.

    Winsorization caps extreme values at specified percentiles rather than removing them.
    Only applies to numerical columns with more than 2 unique values (excludes binary).
    Uses the 3-sigma rule: clips values beyond mean ± 3*std to the 5th/95th percentile.

    Args:
        dataframe: Input DataFrame with numerical features
        lower_quantile: Lower percentile for clipping (default: 0.05)
        upper_quantile: Upper percentile for clipping (default: 0.95)

    Returns:
        DataFrame with winsorized numerical columns

    Example:
        >>> df = winsorize_dataframe(df, lower_quantile=0.05, upper_quantile=0.95)
    """
    # Select numerical columns with more than 2 unique values
    # (excludes binary features like has_superstructure_adobe_mud)
    num_cols = [
        col for col in dataframe.select_dtypes(include=[np.number]).columns
        if dataframe[col].nunique() > 2
    ]

    # Get summary statistics for all numerical columns
    description = dataframe.describe()

    # Apply winsorization to columns with extreme outliers
    for col in description.columns:
        mean_ = description[col].loc['mean']
        std_ = description[col].loc['std']

        # Check for extreme positive outliers (beyond mean + 3*std)
        if dataframe[col].max() > mean_ + 3 * std_:
            # Clip lower extreme values at 5th percentile
            p5 = dataframe[col].quantile(lower_quantile)
            dataframe[col] = dataframe[col].clip(lower=p5)

        # Check for extreme negative outliers (below mean - 3*std)
        if dataframe[col].min() < mean_ - 3 * std_:
            # Clip upper extreme values at 95th percentile
            p95 = dataframe[col].quantile(upper_quantile)
            dataframe[col] = dataframe[col].clip(upper=p95)

    return dataframe


# ==============================================================================
# Feature Selection
# ==============================================================================

def drop_columns(dataframe: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Remove specified columns from DataFrame.

    Used to drop non-predictive features, redundant columns, or those causing
    data leakage (e.g., building_id, secondary_use features).

    Args:
        dataframe: Input DataFrame
        columns_to_drop: List of column names to remove

    Returns:
        DataFrame with specified columns removed

    Example:
        >>> df = drop_columns(df, ['building_id', 'count_floors_pre_eq'])
    """
    return dataframe.drop(columns=columns_to_drop, errors='ignore')


# ==============================================================================
# Feature Encoding
# ==============================================================================

def encode_features(
        dataframe: pd.DataFrame,
        y: np.ndarray,
        dataframe_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical features using appropriate strategies.

    Encoding strategy:
    - OneHot Encoding: For geographic IDs (low-to-medium cardinality)
    - Target Encoding: For other categoricals (captures relationship with target)

    The preprocessor is fitted ONLY on training data to prevent data leakage.
    Test data is transformed using the fitted encoder.

    Args:
        dataframe: Training feature DataFrame
        y: Training labels (encoded as 0, 1, 2)
        dataframe_test: Test feature DataFrame

    Returns:
        Tuple of (X_train_encoded, X_test_encoded) as numpy arrays

    Example:
        >>> X_train_enc, X_test_enc = encode_features(train_X, y_train, test_X)
    """
    # Low-medium cardinality geographic features
    # OneHot encoding creates interpretable binary features
    cols_onehot = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

    # Higher cardinality categorical features
    # Target encoding captures relationship with damage grade
    cols_target = [
        "land_surface_condition",
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position"
    ]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cols_onehot),
            ("target", TargetEncoder(smooth="auto"), cols_target)
        ],
        remainder="passthrough"  # Keep all other columns unchanged
    )

    # Fit on training data and transform both train and test
    # Note: TargetEncoder needs y to learn target-based encodings
    x_train_encoded = preprocessor.fit_transform(dataframe, y)
    x_test_encoded = preprocessor.transform(dataframe_test)

    return x_train_encoded, x_test_encoded


# ==============================================================================
# Label Processing
# ==============================================================================

def encode_labels(labels: pd.Series) -> np.ndarray:
    """
    Convert damage grade labels from 1-3 scale to 0-2 scale.

    Many ML algorithms (including XGBoost with multi:softprob) expect
    class labels to start at 0. This function converts:
    - Grade 1 (low damage) → 0
    - Grade 2 (medium damage) → 1
    - Grade 3 (high damage) → 2

    Args:
        labels: Series with damage grades (1, 2, 3)

    Returns:
        Numpy array with encoded labels (0, 1, 2)

    Example:
        >>> y_encoded = encode_labels(train_y["damage_grade"])
        >>> # Original: [1, 2, 3, 1, 3]
        >>> # Encoded:  [0, 1, 2, 0, 2]
    """
    return labels.to_numpy() - 1

# ==============================================================================
# Deprecated/Unused Functions
# ==============================================================================

# The following functions were replaced by the ColumnTransformer approach
# in encode_features() for better pipeline integration

# def target_encode_feature(feature_to_encode, labels):
#     """
#     Apply target encoding to a single feature.
#     DEPRECATED: Use encode_features() instead for full pipeline.
#     """
#     encoder = TargetEncoder(smooth="auto")
#     return encoder.fit_transform(feature_to_encode, labels)

# def onehot_encoder(feature_to_encode):
#     """
#     Apply one-hot encoding to a single feature.
#     DEPRECATED: Use encode_features() instead for full pipeline.
#     """
#     encoder = OneHotEncoder(handle_unknown="ignore")
#     return encoder.fit_transform(feature_to_encode)