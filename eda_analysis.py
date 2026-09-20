from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_trains.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}\n"
        "Run prepare_data.py first."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== RAILWISE AI EDA ==========")
print("Dataset loaded successfully!")

# Required columns from preprocessing
required = [
    "Train_No",
    "Origin_Station",
    "Destination_Station",
    "Total_Distance",
    "Number_of_Stops",
    "Journey_Duration_Minutes",
    "Journey_Duration_Hours",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Numeric conversion
numeric_cols = [
    "Train_No",
    "Total_Distance",
    "Number_of_Stops",
    "Journey_Duration_Minutes",
    "Journey_Duration_Hours",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== TASK 1.1 ==========")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])
print("Unique train numbers:", df["Train_No"].nunique())

print("\nColumn names:")
print(df.columns.tolist())

# Save dataset overview
overview = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Columns",
        "Unique Train Numbers",
        "Exact Duplicate Rows",
        "Total Missing Cells",
    ],
    "Value": [
        len(df),
        len(df.columns),
        df["Train_No"].nunique(),
        df.duplicated().sum(),
        df.isnull().sum().sum(),
    ],
})

overview.to_csv(
    OUTPUT_DIR / "level1_dataset_overview.csv",
    index=False
)


print("\n========== TASK 1.2 ==========")

train_table = df[
    [
        "Train_No",
        "Origin_Station",
        "Destination_Station",
        "Total_Distance",
        "Number_of_Stops",
        "Journey_Duration_Hours",
    ]
].copy()

train_table = train_table.sort_values("Train_No")

print(train_table.head(10).to_string(index=False))

train_table.to_csv(
    OUTPUT_DIR / "level1_train_wise_routes.csv",
    index=False
)


print("\n========== TASK 1.3 ==========")

stats_columns = [
    "Total_Distance",
    "Number_of_Stops",
]

statistics = df[stats_columns].describe().T

print(statistics)

statistics.to_csv(
    OUTPUT_DIR / "level1_distance_stop_statistics.csv"
)


print("\n========== TASK 1.4 ==========")

missing_values = df.isnull().sum()

duplicate_count = df.duplicated().sum()

print("\nMissing values by column:")
print(missing_values)

print("\nExact duplicate rows:", duplicate_count)

# Potential data-quality checks
quality_checks = pd.DataFrame({
    "Check": [
        "Missing values",
        "Exact duplicate rows",
        "Missing train number",
        "Missing origin station",
        "Missing destination station",
        "Non-positive distance",
        "Negative stop count",
        "Non-positive journey duration",
        "Journey duration above 7 days",
    ],
    "Count": [
        int(df.isnull().sum().sum()),
        int(duplicate_count),
        int(df["Train_No"].isnull().sum()),
        int(df["Origin_Station"].isnull().sum()),
        int(df["Destination_Station"].isnull().sum()),
        int((df["Total_Distance"] <= 0).sum()),
        int((df["Number_of_Stops"] < 0).sum()),
        int((df["Journey_Duration_Minutes"] <= 0).sum()),
        int(
            (
                df["Journey_Duration_Minutes"]
                > 7 * 24 * 60
            ).sum()
        ),
    ],
})

print("\nQuality checks:")
print(quality_checks.to_string(index=False))

quality_checks.to_csv(
    OUTPUT_DIR / "level1_quality_checks.csv",
    index=False
)


# Remove rows with invalid values for plotting
plot_df = df.dropna(
    subset=[
        "Total_Distance",
        "Number_of_Stops",
        "Journey_Duration_Hours",
    ]
).copy()


print("\n========== TASK 3.1 ==========")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_df,
    x="Total_Distance",
    y="Journey_Duration_Hours",
    alpha=0.5
)

sns.regplot(
    data=plot_df,
    x="Total_Distance",
    y="Journey_Duration_Hours",
    scatter=False,
    color="red"
)

plt.title("Distance vs Journey Duration")
plt.xlabel("Total Distance")
plt.ylabel("Journey Duration (Hours)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level3_distance_vs_duration.png",
    dpi=300
)

plt.close()


print("\n========== TASK 3.2 ==========")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_df,
    x="Number_of_Stops",
    y="Journey_Duration_Hours",
    alpha=0.5
)

plt.title("Number of Stops vs Journey Duration")
plt.xlabel("Number of Stops")
plt.ylabel("Journey Duration (Hours)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level3_stops_vs_duration.png",
    dpi=300
)

plt.close()

print("\n========== TASK 3.3 ==========")

correlation_columns = [
    "Total_Distance",
    "Total_Stations",
    "Number_of_Stops",
    "Departure_Minutes",
    "Journey_Duration_Minutes",
    "Journey_Duration_Hours",
]

# Use only columns that exist in the CSV
correlation_columns = [
    col for col in correlation_columns
    if col in df.columns
]

correlation_matrix = df[
    correlation_columns
].corr(numeric_only=True)

print(correlation_matrix)

correlation_matrix.to_csv(
    OUTPUT_DIR / "level3_correlation_matrix.csv"
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - RailWise AI")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "level3_correlation_heatmap.png",
    dpi=300
)

plt.close()


print("\n========== TASK 3.4 ==========")

pivot_table = pd.pivot_table(
    df,
    index="Train_No",
    values="Number_of_Stops",
    aggfunc=["count", "mean", "min", "max"]
)

print(pivot_table.head(10))

pivot_table.to_csv(
    OUTPUT_DIR / "level3_stops_pivot_table.csv"
)

top_duration = (
    df[
        [
            "Train_No",
            "Origin_Station",
            "Destination_Station",
            "Journey_Duration_Hours",
        ]
    ]
    .sort_values(
        "Journey_Duration_Hours",
        ascending=False
    )
    .head(10)
)

top_duration.to_csv(
    OUTPUT_DIR / "top_10_longest_journeys.csv",
    index=False
)

print("\nTop 10 longest reconstructed journeys:")
print(top_duration.to_string(index=False))

print("\n========== EDA COMPLETED ==========")
print("All reports and charts saved in:")
print(OUTPUT_DIR)