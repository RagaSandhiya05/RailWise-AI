from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="RailWise AI",
    page_icon="🚆",
    layout="centered",
)

st.title("🚆 RailWise AI")
st.subheader("Train Journey Duration Prediction")

st.write(
    "Enter journey details to estimate the "
    "scheduled train journey duration."
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

METRICS_PATH = (
    BASE_DIR / "outputs" / "level5_model_comparison.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR / "outputs" / "level5_predictions.csv"
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. "
        "Please run compare_models.py first."
    )
    st.stop()

try:
    saved_model = joblib.load(MODEL_PATH)

except Exception as error:
    st.error(
        f"Could not load the saved model: {error}"
    )
    st.stop()


# Extract the model package
if not isinstance(saved_model, dict):
    st.error(
        "The saved file does not contain the expected "
        "model package. Please rerun compare_models.py."
    )
    st.stop()

model = saved_model.get("model")
features = saved_model.get("features")
model_name = saved_model.get(
    "model_name",
    "Trained Model"
)

if model is None or not features:
    st.error(
        "The saved model package is missing its model "
        "or feature names. Rerun compare_models.py."
    )
    st.stop()

st.success(
    f"Loaded model: {model_name}"
)

st.divider()
st.header("Enter Journey Details")

with st.form("journey_prediction_form"):

    distance = st.number_input(
        "Total journey distance (km)",
        min_value=1.0,
        max_value=10000.0,
        value=500.0,
        step=10.0,
    )

    stops = st.number_input(
        "Number of intermediate stops",
        min_value=0,
        max_value=200,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button(
        "Predict Journey Duration"
    )

if submitted:

    input_values = {
        "Total_Distance": distance,
        "Number_of_Stops": stops,
    }

    try:
        # Keep input columns in the trained model's order.
        input_data = pd.DataFrame([
            {
                feature: input_values[feature]
                for feature in features
            }
        ])

        prediction = float(
            model.predict(input_data)[0]
        )

        if prediction < 0:
            st.warning(
                "The model returned a negative duration. "
                "Please check your inputs and model."
            )

        else:
            total_minutes = int(round(prediction))

            hours = total_minutes // 60
            minutes = total_minutes % 60

            st.success(
                "Prediction completed successfully!"
            )

            st.metric(
                label="Estimated Journey Duration",
                value=f"{prediction:,.1f} minutes",
            )

            st.info(
                f"Approximately **{hours} hours "
                f"and {minutes} minutes**."
            )

            st.write("### Input Summary")

            summary = pd.DataFrame({
                "Feature": [
                    "Journey Distance",
                    "Intermediate Stops",
                ],
                "Value": [
                    f"{distance:,.1f} km",
                    int(stops),
                ],
            })

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "This is an estimate based on scheduled "
                "train data. It is not a live train status "
                "or delay prediction."
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )

st.divider()
st.header("📊 Level 5: Model Comparison")

if METRICS_PATH.exists():

    metrics_df = pd.read_csv(
        METRICS_PATH
    )

    st.dataframe(
        metrics_df.round(4),
        hide_index=True,
        use_container_width=True,
    )

    if "Model" in metrics_df.columns:

        chart_metric = st.selectbox(
            "Choose metric to compare",
            ["MAE", "RMSE", "R2"],
        )

        if chart_metric in metrics_df.columns:

            chart_df = metrics_df[
                ["Model", chart_metric]
            ].set_index("Model")

            st.bar_chart(
                chart_df
            )

else:
    st.info(
        "Model comparison file not found. "
        "Run compare_models.py first."
    )

st.divider()
st.header("📈 Actual vs Predicted Results")

if PREDICTIONS_PATH.exists():

    results = pd.read_csv(
        PREDICTIONS_PATH
    )

    actual_col = "Actual_Duration_Minutes"

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    prediction_col = (
        f"{safe_name}_prediction"
    )

    if (
        actual_col in results.columns
        and prediction_col in results.columns
    ):

        st.write(
            f"Showing test results for **{model_name}**."
        )

        plot_data = results[
            [actual_col, prediction_col]
        ].dropna()

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            plot_data[actual_col],
            plot_data[prediction_col],
            alpha=0.5,
        )

        lower = min(
            plot_data[actual_col].min(),
            plot_data[prediction_col].min(),
        )

        upper = max(
            plot_data[actual_col].max(),
            plot_data[prediction_col].max(),
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
        )

        ax.set_xlabel(
            "Actual Duration (Minutes)"
        )

        ax.set_ylabel(
            "Predicted Duration (Minutes)"
        )

        ax.set_title(
            f"Actual vs Predicted - {model_name}"
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write("### Test Prediction Preview")

        st.dataframe(
            results[
                [
                    actual_col,
                    prediction_col,
                ]
            ].head(10),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.warning(
            "The expected actual/prediction columns "
            "were not found. Please rerun compare_models.py."
        )

else:
    st.info(
        "Prediction results not found. "
        "Run compare_models.py first."
    )

st.divider()

st.caption(
    "RailWise AI | Train Journey Duration Prediction"
)