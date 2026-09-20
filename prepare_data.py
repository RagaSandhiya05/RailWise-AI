from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Dataset1.csv"
OUTPUT_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "SN",
    "Train_No",
    "Station_Code",
    "Station_Name",
    "Arrival_time",
    "Departure_Time",
    "Distance"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\n========== DATASET OVERVIEW ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique trains:", df["Train_No"].nunique())
print("Missing values:", df.isnull().sum().sum())
print("Exact duplicate rows:", df.duplicated().sum())

quality_records = []

# Missing values by column
for column in df.columns:
    missing_count = int(df[column].isnull().sum())

    if missing_count > 0:
        quality_records.append({
            "Train_No": "ALL",
            "Issue": f"{column}: {missing_count} missing values"
        })

# Exact duplicate rows
duplicate_count = int(df.duplicated().sum())

if duplicate_count > 0:
    quality_records.append({
        "Train_No": "ALL",
        "Issue": f"{duplicate_count} exact duplicate rows"
    })

# Remove exact duplicates
df = df.drop_duplicates().copy()


df["SN"] = pd.to_numeric(
    df["SN"], errors="coerce"
)

df["Train_No"] = pd.to_numeric(
    df["Train_No"], errors="coerce"
)

df["Distance"] = pd.to_numeric(
    df["Distance"], errors="coerce"
)

df["Arrival_time"] = (
    df["Arrival_time"]
    .astype("string")
    .str.strip()
)

df["Departure_Time"] = (
    df["Departure_Time"]
    .astype("string")
    .str.strip()
)

# Check rows with missing required fields
required_data = [
    "Train_No",
    "SN",
    "Station_Code",
    "Station_Name",
    "Distance"
]

invalid_required = df[required_data].isnull().any(axis=1)

invalid_required_count = int(invalid_required.sum())

if invalid_required_count > 0:
    quality_records.append({
        "Train_No": "ALL",
        "Issue": (
            f"{invalid_required_count} rows have missing "
            "or invalid required fields"
        )
    })

# Remove rows missing required fields
df = df.dropna(
    subset=required_data
).copy()


df = df.sort_values(
    ["Train_No", "SN"]
).reset_index(drop=True)

def time_to_minutes(value):
    """
    Convert a valid HH:MM or HH:MM:SS time
    into minutes after midnight.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    try:
        parsed = pd.to_timedelta(value)
        minutes = parsed.total_seconds() / 60

        if minutes < 0 or minutes >= 24 * 60:
            return None

        return int(minutes)

    except (ValueError, TypeError):
        return None

train_records = []

for train_no, group in df.groupby(
    "Train_No", sort=False
):

    group = (
        group.sort_values("SN")
        .reset_index(drop=True)
    )

    issues = []

    if len(group) < 2:
        issues.append(
            "Fewer than 2 station records"
        )

        quality_records.append({
            "Train_No": train_no,
            "Issue": "; ".join(issues)
        })

        continue

    if group["SN"].duplicated().any():
        issues.append(
            "Duplicate station sequence numbers"
        )


    if group["Station_Code"].nunique() < 2:
        issues.append(
            "Fewer than 2 unique station codes"
        )

    origin = group.iloc[0]
    destination = group.iloc[-1]

    distances = group["Distance"].tolist()

    if any(
        distances[i] < distances[i - 1]
        for i in range(1, len(distances))
    ):
        issues.append(
            "Distance decreases along route"
        )

    total_distance = (
        destination["Distance"]
        - origin["Distance"]
    )

    if (
        pd.isna(total_distance)
        or total_distance <= 0
    ):
        issues.append(
            "Invalid total distance"
        )

    start_minutes = time_to_minutes(
        origin["Departure_Time"]
    )

    if start_minutes is None:
        issues.append(
            "Invalid origin departure time"
        )

        quality_records.append({
            "Train_No": train_no,
            "Issue": "; ".join(issues)
        })

        continue

    previous_minutes = start_minutes
    final_arrival = None
    invalid_time = False

    for i in range(1, len(group)):

        station = group.iloc[i]

        arrival = time_to_minutes(
            station["Arrival_time"]
        )

        if arrival is None:
            issues.append(
                f"Invalid arrival time at SN {station['SN']}"
            )

            invalid_time = True
            break

        # Handle overnight arrival
        while arrival < previous_minutes:
            arrival += 24 * 60

        final_arrival = arrival
        previous_minutes = arrival

        # No need to process final station departure
        if i == len(group) - 1:
            break

        departure = time_to_minutes(
            station["Departure_Time"]
        )

        if departure is None:
            issues.append(
                f"Invalid departure time at SN {station['SN']}"
            )

            invalid_time = True
            break

        # Handle overnight departure
        while departure < previous_minutes:
            departure += 24 * 60

        previous_minutes = departure

    if invalid_time or final_arrival is None:

        quality_records.append({
            "Train_No": train_no,
            "Issue": "; ".join(issues)
        })

        continue

    duration_minutes = (
        final_arrival - start_minutes
    )

    if duration_minutes <= 0:
        issues.append(
            "Non-positive journey duration"
        )

    if duration_minutes > 7 * 24 * 60:
        issues.append(
            "Duration exceeds 7 days"
        )

    if issues:
        quality_records.append({
            "Train_No": train_no,
            "Issue": "; ".join(issues)
        })

    if (
        duration_minutes <= 0
        or duration_minutes > 7 * 24 * 60
        or pd.isna(total_distance)
        or total_distance <= 0
    ):
        continue

    train_records.append({

        "Train_No": int(train_no),

        "Origin_Code": origin["Station_Code"],
        "Origin_Station": origin["Station_Name"],

        "Destination_Code": destination["Station_Code"],
        "Destination_Station": destination["Station_Name"],

        "Total_Distance": float(total_distance),

        "Total_Stations": len(group),

        "Number_of_Stops": max(
            len(group) - 2, 0
        ),

        "Departure_Minutes": start_minutes,

        "Arrival_Minutes": (
            final_arrival % (24 * 60)
        ),

        "Journey_Duration_Minutes": duration_minutes,

        "Journey_Duration_Hours": round(
            duration_minutes / 60, 2
        )
    })


train_df = pd.DataFrame(train_records)

if quality_records:
    quality_df = pd.DataFrame(quality_records)
else:
    quality_df = pd.DataFrame([
        {
            "Train_No": "N/A",
            "Issue": (
                "No issues flagged by the "
                "implemented quality checks"
            )
        }
    ])


clean_path = OUTPUT_DIR / "cleaned_trains.csv"

quality_path = OUTPUT_DIR / "quality_report.csv"

train_df.to_csv(
    clean_path,
    index=False
)

quality_df.to_csv(
    quality_path,
    index=False
)

print("\n========== TRAIN-WISE DATA ==========")

print("Valid train records:", len(train_df))

print(
    "Quality report entries:",
    len(quality_df)
)

print("\n========== QUALITY REPORT ==========")

print(quality_df.to_string(index=False))

if not train_df.empty:

    print("\n========== DURATION STATISTICS ==========")

    print(
        train_df[
            "Journey_Duration_Hours"
        ].describe()
    )

    print("\n========== SAMPLE TRAIN RECORDS ==========")

    print(
        train_df.head(10).to_string(index=False)
    )

print("\n========== SAVED FILES ==========")

print("Cleaned data:", clean_path)

print("Quality report:", quality_path)

print("\nData preparation completed successfully!")