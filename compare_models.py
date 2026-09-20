from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

TARGET = "Journey_Duration_Minutes"

FEATURES_BASIC = [
    "Total_Distance"
]

FEATURES_IMPROVED = [
    "Total_Distance",
    "Number_of_Stops",
]

required = FEATURES_IMPROVED + [TARGET]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

for col in required:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required
).copy()

df = df[
    (df["Total_Distance"] > 0)
    & (df["Number_of_Stops"] >= 0)
    & (df[TARGET] > 0)
].copy()

if len(df) < 10:
    raise ValueError(
        "Not enough valid records to compare models."
    )

print("Valid records:", len(df))

X = df[FEATURES_IMPROVED]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

models = {
    "Basic Linear Regression": {
        "model": LinearRegression(),
        "features": FEATURES_BASIC,
    },

    "Multiple Linear Regression": {
        "model": LinearRegression(),
        "features": FEATURES_IMPROVED,
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "features": FEATURES_IMPROVED,
    },
}

comparison_results = []
predictions = {}

for name, config in models.items():

    model = config["model"]
    features = config["features"]

    model.fit(
        X_train[features],
        y_train,
    )

    y_pred = model.predict(
        X_test[features]
    )

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    comparison_results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    predictions[name] = y_pred

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R2:   {r2:.4f}")

comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df.sort_values(
    "MAE",
    ascending=True,
).reset_index(drop=True)

comparison_path = (
    OUTPUT_DIR / "level5_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print("\n========== MODEL COMPARISON ==========")
print(
    comparison_df.to_string(index=False)
)


best_model_name = comparison_df.iloc[0]["Model"]

best_config = models[best_model_name]

best_model = best_config["model"]
best_features = best_config["features"]

print("\nSelected model by lowest test MAE:")
print(best_model_name)

model_package = {
    "model": best_model,
    "features": best_features,
    "target": TARGET,
    "model_name": best_model_name,

    "metrics": {
        row["Model"]: {
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
        }
        for _, row in comparison_df.iterrows()
    },
}

model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(
    model_package,
    model_path,
)

print("\nModel saved successfully!")
print("Model:", best_model_name)
print("Path:", model_path)

prediction_df = X_test.copy()

prediction_df["Actual_Duration_Minutes"] = (
    y_test.to_numpy()
)

for name, y_pred in predictions.items():

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    prediction_df[
        f"{safe_name}_prediction"
    ] = y_pred

prediction_path = (
    OUTPUT_DIR / "level5_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)

print("Predictions saved:", prediction_path)

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Model"],
    comparison_df["MAE"],
)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (Minutes)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level5_mae_comparison.png",
    dpi=300,
)

plt.close()

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Model"],
    comparison_df["RMSE"],
)

plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error (Minutes)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level5_rmse_comparison.png",
    dpi=300,
)

plt.close()

best_predictions = predictions[
    best_model_name
]

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6,
)

lower = min(
    y_test.min(),
    best_predictions.min(),
)

upper = max(
    y_test.max(),
    best_predictions.max(),
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
)

plt.xlabel("Actual Duration (Minutes)")
plt.ylabel("Predicted Duration (Minutes)")

plt.title(
    f"Actual vs Predicted - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level5_best_model_actual_vs_predicted.png",
    dpi=300,
)

plt.close()

print("\n========== LEVEL 5 COMPLETED ==========")
print("Selected model:", best_model_name)
print("Saved model:", model_path)
print("Saved comparison:", comparison_path)
print("Saved predictions:", prediction_path)
print("All charts saved in:", OUTPUT_DIR)