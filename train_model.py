from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_trains.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "cleaned_trains.csv not found. "
        "Run prepare_data.py first."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== RAILWISE AI ==========")
print("Level 4: Model Training and Evaluation")
print("Original dataset shape:", df.shape)

FEATURES = [
    "Total_Distance",
    "Number_of_Stops",
]

TARGET = "Journey_Duration_Minutes"

required = FEATURES + [TARGET]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert to numeric
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinity with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Keep complete rows
df = df.dropna(subset=required).copy()

# Remove impossible values for this initial model
df = df[
    (df["Total_Distance"] > 0)
    & (df["Number_of_Stops"] >= 0)
    & (df[TARGET] > 0)
].copy()

if len(df) < 10:
    raise ValueError(
        "Not enough valid records to train and test."
    )

print("Valid records:", len(df))

X = df[FEATURES]
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed!")

print("\nModel coefficients:")
for feature, coefficient in zip(
    FEATURES, model.coef_
):
    print(f"{feature}: {coefficient:.4f}")

print("Intercept:", round(model.intercept_, 4))

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE : {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R²  : {r2:.4f}")

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
})

metrics.to_csv(
    OUTPUT_DIR / "level4_model_metrics.csv",
    index=False
)

results = X_test.copy()

results["Actual_Duration_Minutes"] = y_test
results["Predicted_Duration_Minutes"] = y_pred

results["Absolute_Error_Minutes"] = np.abs(
    y_test.to_numpy() - y_pred
)

results.to_csv(
    OUTPUT_DIR / "level4_predictions.csv",
    index=False
)

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Journey Duration (Minutes)")
plt.ylabel("Predicted Journey Duration (Minutes)")
plt.title("Actual vs Predicted - RailWise AI")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level4_actual_vs_predicted.png",
    dpi=300
)

plt.close()

model_package = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    },
}

joblib.dump(
    model_package,
    MODEL_DIR / "linear_regression_model.pkl"
)


example = pd.DataFrame({
    "Total_Distance": [500],
    "Number_of_Stops": [10],
})

example_prediction = model.predict(example)[0]

print("\n========== EXAMPLE PREDICTION ==========")
print("Example distance: 500")
print("Example stops: 10")
print(
    f"Predicted duration: "
    f"{example_prediction:.2f} minutes"
)

print("\nModel saved in:", MODEL_DIR)
print("Evaluation outputs saved in:", OUTPUT_DIR)
print("\nLEVEL 4 COMPLETED!")