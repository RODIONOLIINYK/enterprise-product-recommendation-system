import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    PROJECT_ROOT / "artifacts/catboost/test_predictions.csv"
)

calibration_features = (
    df[["prediction", "previous_paid_purchase_count", "rank"]]
)

calibration_labels = df["label"].astype(int)

calibrator = LogisticRegression(max_iter=8000)
calibrator.fit(
    calibration_features,
    calibration_labels,
)

print('iterations: ', calibrator.n_iter_)

df["purchase_probability"] = calibrator.predict_proba(
    calibration_features
)[:, 1]

joblib.dump(
    calibrator,
    PROJECT_ROOT / "models/purchase_probability_calibrator.joblib",
)

print(df[[
    "label",
    "rank",
    "previous_paid_purchase_count",
    "prediction",
    "purchase_probability",
]].head(30))