
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "prepared_data"

TARGET_COLUMN = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42

COLUMNS_TO_DROP = [
    "Unnamed: 0",
    "CustomerID",
]

# 
CATEGORY_CORRECTIONS = {
    "Gender": {
        "Fe Male": "Female",
    },
    "Occupation": {
        "Free Lancer": "Freelancer",
    }
}

EXPECTED_TARGET_VALUES = {0, 1}

# Load and validate the source dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Source dataset was not found: {DATASET_PATH}"
    )

data = pd.read_csv(DATASET_PATH)

print(f"Original dataset shape: {data.shape}")

if TARGET_COLUMN not in data.columns:
    raise KeyError(
        f"Target column '{TARGET_COLUMN}' is missing."
    )

# Remove non-predictive columns
data = data.drop(
    columns=COLUMNS_TO_DROP,
    errors="ignore",
)

# Clean categorical values
categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    # Strip unnecessary surrounding whitespace while
    # preserving genuine missing values.
    data[column] = data[column].str.strip()

    # Convert blank strings to missing values.
    data[column] = data[column].replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

for column, replacements in CATEGORY_CORRECTIONS.items():
    if column not in data.columns:
        raise KeyError(
            f"Expected categorical column '{column}' is missing."
        )

    data[column] = data[column].replace(
        replacements
    )

# Remove duplicate records
duplicate_count = int(data.duplicated().sum())

data = (
    data
    .drop_duplicates()
    .reset_index(drop=True)
)

# Validate target
if data[TARGET_COLUMN].isna().any():
    raise ValueError(
        f"Missing values found in target column: "
        f"{TARGET_COLUMN}"
    )

actual_target_values = set(
    data[TARGET_COLUMN].unique()
)

if not actual_target_values.issubset(
    EXPECTED_TARGET_VALUES
):
    raise ValueError(
        f"Unexpected target values found: "
        f"{sorted(actual_target_values)}"
    )

if data[TARGET_COLUMN].nunique() != 2:
    raise ValueError(
        "The target must contain both classes 0 and 1."
    )

data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(
    "int64"
)

# Display cleaned categorical values
print("\nCleaned categorical values:")

for column in categorical_columns:
    print(
        f"{column}: "
        f"{sorted(data[column].dropna().unique().tolist())}"
    )

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
        "\nThese values will be handled by imputers fitted "
        "only on the training data inside the model pipeline."
    )

# Split features and target
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Create train and test datasets
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

# Save prepared datasets
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_data.to_csv(
    train_path,
    index=False,
)

test_data.to_csv(
    test_path,
    index=False,
)

# Display preparation summary
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
