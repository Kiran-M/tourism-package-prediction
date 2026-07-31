
from pathlib import Path

import pandas as pd

# Define the project dataset and target column
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "tourism.csv"
TARGET_COLUMN = "ProdTaken"

# Define the columns required for model development
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
]

def register_dataset():
    """Load and validate the tourism dataset."""

    # Verify that the source dataset is available
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load the source CSV and provide meaningful parsing errors
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

    # Remove the exported DataFrame index when present
    data = data.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )

    # Ensure that usable records remain after loading
    if data.empty:
        raise ValueError(
            "The dataset contains no usable records."
        )

    # Validate the required dataset schema
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing expected columns: "
            f"{missing_columns}"
        )

    # Validate that the target contains usable values
    if data[TARGET_COLUMN].isna().all():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains no usable values."
        )

    # Calculate the initial data-quality summary
    duplicate_count = int(data.duplicated().sum())
    missing_value_count = int(data.isna().sum().sum())

    # Display the registered dataset summary
    print("Dataset registered and validated successfully.")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print(f"Column names: {list(data.columns)}")
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Total missing values: {missing_value_count}")

    print("\nTarget distribution:")
    print(
        data[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    return data

def main():
    """Run dataset registration."""

    try:
        register_dataset()
    except Exception as error:
        print(f"\nData registration failed: {error}")
        raise

if __name__ == "__main__":
    main()
