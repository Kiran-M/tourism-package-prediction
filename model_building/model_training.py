
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]

PREPARED_DATA_DIR = Path(
    os.getenv(
        "PREPARED_DATA_DIR",
        str(PROJECT_DIR / "prepared_data"),
    )
)

TRAIN_PATH = PREPARED_DATA_DIR / "train.csv"
TEST_PATH = PREPARED_DATA_DIR / "test.csv"

MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
TUNING_RESULTS_PATH = MODEL_DIR / "tuning_results.csv"
CLASSIFICATION_REPORT_PATH = (
    MODEL_DIR / "classification_report.png"
)
CONFUSION_MATRIX_PATH = (
    MODEL_DIR / "confusion_matrix.png"
)

TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42
EXPERIMENT_NAME = "tourism-production-experimentation"

# Configure production experiment tracking
DEFAULT_TRACKING_DIR = (
    PROJECT_DIR
    / "production_tracking"
    / "mlruns"
)

DEFAULT_TRACKING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    DEFAULT_TRACKING_DIR.resolve().as_uri(),
)

mlflow.set_tracking_uri(TRACKING_URI)

experiment = mlflow.set_experiment(
    EXPERIMENT_NAME
)

if mlflow.active_run() is not None:
    mlflow.end_run()

# Validate the prepared datasets
missing_files = [
    str(path)
    for path in (TRAIN_PATH, TEST_PATH)
    if not path.is_file()
]

if missing_files:
    raise FileNotFoundError(
        "Required prepared-data files were not found: "
        + ", ".join(missing_files)
    )

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

for dataset_name, dataset in {
    "training": train_data,
    "testing": test_data,
}.items():
    if TARGET_COLUMN not in dataset.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' is missing "
            f"from the {dataset_name} dataset."
        )

X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]

X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]

print("Training artifact:", TRAIN_PATH)
print("Testing artifact:", TEST_PATH)
print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)

# Define preprocessing
categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_columns,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
)

# Handle class imbalance
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0 or negative_count == 0:
    raise ValueError(
        "Training data must contain both target classes."
    )

scale_pos_weight = negative_count / positive_count

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=1,
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# Define the production hyperparameter search
parameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.10],
    "model__subsample": [0.8, 1.0],
}

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cross_validation,
    n_jobs=-1,
    return_train_score=True,
    verbose=1,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Tune, evaluate and track the production model
with mlflow.start_run(
    run_name="production-xgboost-grid-search"
) as parent_run:

    parent_run_id = parent_run.info.run_id

    mlflow.log_params(
        {
            "algorithm": "XGBoost",
            "selection_metric": "f1",
            "cv_folds": cross_validation.n_splits,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "categorical_features": len(
                categorical_columns
            ),
            "numerical_features": len(
                numerical_columns
            ),
            "scale_pos_weight": scale_pos_weight,
        }
    )

    mlflow.log_dict(
        parameter_grid,
        "tuning/parameter_grid.json",
    )

    grid_search.fit(X_train, y_train)

    tuning_results = pd.DataFrame(
        grid_search.cv_results_
    )

    tuning_results.to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )

    mlflow.log_artifact(
        str(TUNING_RESULTS_PATH),
        artifact_path="tuning",
    )

    # Log each tuning trial separately
    for trial_number, row in tuning_results.iterrows():

        with mlflow.start_run(
            run_name=f"production-trial-{trial_number + 1}",
            nested=True,
        ):

            mlflow.log_params(row["params"])

            mlflow.log_metrics(
                {
                    "mean_cv_f1": float(
                        row["mean_test_score"]
                    ),
                    "std_cv_f1": float(
                        row["std_test_score"]
                    ),
                    "mean_train_f1": float(
                        row["mean_train_score"]
                    ),
                }
            )

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "test_precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }

    best_parameters = {
        name.replace("model__", ""): value
        for name, value in grid_search.best_params_.items()
    }

    mlflow.log_params(
        {
            f"best_{name}": value
            for name, value in best_parameters.items()
        }
    )

    mlflow.log_metric(
        "best_cv_f1",
        float(grid_search.best_score_),
    )

    mlflow.log_metrics(metrics)

    # Create the classification report
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    mlflow.log_dict(
        report,
        "evaluation/classification_report.json",
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .drop(index="accuracy")
        .drop(columns="support")
        .rename(
            index={
                "0": "Not Purchased",
                "1": "Purchased",
            }
        )
    )

    classification_figure, classification_axis = (
        plt.subplots(figsize=(8, 5))
    )

    sns.heatmap(
        report_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=classification_axis,
    )

    classification_axis.set_title(
        "Production Classification Report"
    )
    classification_axis.set_xlabel("Metrics")
    classification_axis.set_ylabel(
        "Classes and Averages"
    )

    classification_figure.tight_layout()

    classification_figure.savefig(
        CLASSIFICATION_REPORT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    mlflow.log_figure(
        classification_figure,
        "evaluation/classification_report.png",
    )

    plt.close(classification_figure)

    # Create the confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    mlflow.log_dict(
        {"confusion_matrix": matrix.tolist()},
        "evaluation/confusion_matrix.json",
    )

    confusion_figure, confusion_axis = (
        plt.subplots(figsize=(6, 5))
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Not Purchased",
            "Purchased",
        ],
        yticklabels=[
            "Not Purchased",
            "Purchased",
        ],
        ax=confusion_axis,
    )

    confusion_axis.set_title(
        "Production Confusion Matrix"
    )
    confusion_axis.set_xlabel("Predicted Class")
    confusion_axis.set_ylabel("Actual Class")

    confusion_figure.tight_layout()

    confusion_figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    mlflow.log_figure(
        confusion_figure,
        "evaluation/confusion_matrix.png",
    )

    plt.close(confusion_figure)

    # Save the production model and metrics
    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    evaluation_summary = {
        "best_parameters": best_parameters,
        "best_cv_f1": float(
            grid_search.best_score_
        ),
        **metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_summary,
            file,
            indent=4,
        )

    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="model",
    )

    mlflow.log_artifact(
        str(METRICS_PATH),
        artifact_path="evaluation",
    )

print("\nProduction experiment completed.")

print("\nBest parameters:")
print(best_parameters)

print("\nBest cross-validation F1:")
print(f"{grid_search.best_score_:.4f}")

print("\nTest metrics:")

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
        target_names=[
            "Not Purchased",
            "Purchased",
        ],
    )
)

print("Confusion matrix:")
print(matrix)

print("\nBest model saved at:", MODEL_PATH)
print("Metrics saved at:", METRICS_PATH)
print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow parent run ID:", parent_run_id)
