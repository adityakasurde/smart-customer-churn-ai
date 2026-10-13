"""Data preprocessing pipeline for Smart Customer Churn Prediction AI.

Includes cleaning, missing-value handling, target encoding,
leakage-free train/test splitting, and Scikit-learn ColumnTransformer construction.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco customer dataset.

    Steps performed:
    1. Deep-copy DataFrame to preserve the raw dataset unchanged.
    2. Convert 'TotalCharges' into numeric float, coercing blank spaces to NaN.
    3. Semantics-based imputation: Customers with tenure == 0 have just signed up
       and have not accumulated any billing history; their TotalCharges are logically 0.0.
    4. Remaining missing values (if any) are kept as NaN so the Scikit-learn Pipeline
       imputes them using training-split statistics only, avoiding data leakage.
    5. Validate and map the binary 'Churn' target ('Yes' -> 1, 'No' -> 0).

    Args:
        df: Input raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    Raises:
        ValueError: If required columns are missing or if target contains invalid labels.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame provided for cleaning is empty or None.")

    cleaned_df = df.copy()

    # 1. Convert TotalCharges to numeric
    if "TotalCharges" in cleaned_df.columns:
        # Strip whitespace and coerce invalid/blank strings to NaN
        if cleaned_df["TotalCharges"].dtype == object or cleaned_df["TotalCharges"].dtype.name in ("string", "str"):
            cleaned_df["TotalCharges"] = pd.to_numeric(
                cleaned_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            )
        else:
            cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

        # 2. Semantic handling: For zero tenure, TotalCharges accrued is 0.0
        if "tenure" in cleaned_df.columns:
            zero_tenure_mask = (cleaned_df["tenure"] == 0) & (cleaned_df["TotalCharges"].isna())
            cleaned_df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # 3. Target label validation & binary conversion
    if TARGET_COLUMN in cleaned_df.columns:
        valid_labels = {"Yes", "No", 1, 0, "1", "0"}
        unique_labels = set(cleaned_df[TARGET_COLUMN].dropna().unique())
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            raise ValueError(
                f"Unexpected labels found in target column '{TARGET_COLUMN}': {invalid_labels}. "
                "Expected values: 'Yes' / 'No' or 1 / 0."
            )

        target_map = {"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}
        cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].map(target_map).astype(int)

    return cleaned_df


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input DataFrame into feature matrix X and target vector y.

    Ensures that the customer identifier and target column are removed from X.

    Args:
        df: Cleaned DataFrame.
        target_col: Target column name.
        id_col: Identifier column name.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)

    Raises:
        ValueError: If target_col is not found in df.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    cols_to_drop = [target_col]
    if id_col in df.columns:
        cols_to_drop.append(id_col)

    X = df.drop(columns=cols_to_drop).copy()
    y = df[target_col].copy()

    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically identify numerical and categorical feature columns.

    Args:
        X: Feature DataFrame (excluding ID and target).

    Returns:
        Tuple[List[str], List[str]]: (numerical_features, categorical_features)
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numerical_features]
    return numerical_features, categorical_features


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset to include in test split (default: 0.20).
        random_state: Seed for reproducibility (default: 42).
        stratify: Whether to preserve target class proportions (default: True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Construct Scikit-learn ColumnTransformer preprocessing pipeline.

    - Numerical features:
        1. SimpleImputer(strategy='median')
        2. StandardScaler()
    - Categorical features:
        1. SimpleImputer(strategy='most_frequent')
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Note: handle_unknown='ignore' guarantees inference requests with unseen
    categories will be safely encoded as all-zeros without throwing errors.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        ColumnTransformer: Configured preprocessor pipeline.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
