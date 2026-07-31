
import pandas as pd

DATASET_PATH = "tourism_project/data/tourism.csv"

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

# Load the dataset
data = pd.read_csv(DATASET_PATH).drop(columns=["Unnamed: 0"], errors="ignore")

# Validate expected columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing expected columns: {missing_columns}"
    )

print("Dataset registered and validated successfully.")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print("Column names:", list(data.columns))
print(f"Duplicate rows: {data.duplicated().sum()}")
print(f"Total missing values: {data.isnull().sum().sum()}")

print("\nTarget distribution:")
print(data["ProdTaken"].value_counts())
