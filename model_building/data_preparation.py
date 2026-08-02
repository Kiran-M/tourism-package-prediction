
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Define project paths and preparation settings
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "prepared_data"

TARGET_COLUMN = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Corrections to category names
CATEGORY_CORRECTIONS = {
    "Gender": {
        "Fe Male": "Female",
    },
    "Occupation": {
        "Free Lancer": "Freelancer",
    },
}

EXPECTED_TARGET_VALUES = {0, 1}

def prepare_data():
    """Clean, validate, split and save the tourism dataset."""

    # Verify that the source dataset exists
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load the source dataset
    try:
        data = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "The dataset is empty."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The dataset is not a valid CSV file."
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Unable to read the dataset: {DATASET_PATH}"
        ) from error

    print(f"Original dataset shape: {data.shape}")

    if data.empty:
        raise ValueError(
            "The dataset contains no usable records."
        )

    if TARGET_COLUMN not in data.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    # Remove only the exported DataFrame index before duplicate detection
    data = data.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )

    # Detect and remove genuine duplicate customer records
    duplicate_count = int(data.duplicated().sum())

    data = (
        data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Remove the customer identifier only after duplicate removal
    data = data.drop(
        columns=["CustomerID"],
        errors="ignore",
    )

    # Identify and clean categorical columns
    categorical_columns = (
        data
        .select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    for column in categorical_columns:
        data[column] = data[column].astype("string").str.strip()

        data[column] = data[column].replace(
            r"^\s*$",
            pd.NA,
            regex=True,
        )

    # Correct known inconsistent categorical values
    for column, replacements in CATEGORY_CORRECTIONS.items():
        if column not in data.columns:
            raise KeyError(
                f"Expected categorical column '{column}' is missing."
            )

        data[column] = data[column].replace(replacements)

    # Validate the target column
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing values found in target column '{TARGET_COLUMN}'."
        )

    actual_target_values = set(
        data[TARGET_COLUMN].unique()
    )

    if not actual_target_values.issubset(EXPECTED_TARGET_VALUES):
        raise ValueError(
            "Unexpected target values found: "
            f"{sorted(actual_target_values)}"
        )

    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "The target column must contain both classes 0 and 1."
        )

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int64")

    # Display cleaned categorical values
    print("\nCleaned categorical values:")

    for column in categorical_columns:
        values = sorted(
            data[column]
            .dropna()
            .unique()
            .tolist()
        )

        print(f"{column}: {values}")

    # Report missing feature values
    feature_missing_values = (
        data
        .drop(columns=[TARGET_COLUMN])
        .isna()
        .sum()
    )

    feature_missing_values = feature_missing_values[
        feature_missing_values > 0
    ]

    print("\nMissing feature values before model preprocessing:")

    if feature_missing_values.empty:
        print("None")
    else:
        print(feature_missing_values.to_string())
        print(
            "\nMissing feature values will be handled by imputers "
            "fitted only on the training data."
        )

    # Separate predictors and target
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # Create a reproducible stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_data = pd.concat(
        [
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
        ],
        axis=1,
    )

    test_data = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
        ],
        axis=1,
    )

    # Create the output directory and save prepared datasets
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Display the preparation summary
    print("\nData preparation completed successfully.")
    print(f"Duplicate records removed: {duplicate_count}")
    print(f"Cleaned dataset shape: {data.shape}")
    print(
        f"Training dataset: {train_data.shape}, "
        f"Path: {train_path}"
    )
    print(
        f"Testing dataset: {test_data.shape}, "
        f"Path: {test_path}"
    )

    print("\nTraining target distribution:")
    print(
        train_data[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    print("\nTesting target distribution:")
    print(
        test_data[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    return train_path, test_path

def main():
    """Run data preparation."""

    try:
        prepare_data()
    except Exception as error:
        print(f"\nData preparation failed: {error}")
        raise

if __name__ == "__main__":
    main()
