# 🚆 RailWise AI — Train Journey Duration Prediction

RailWise AI is a machine learning project focused on predicting train journey duration using data analysis, data cleaning, feature engineering, and model comparison. It explores relationships between journey distance, intermediate stops, and travel time, evaluates multiple regression models, and provides an interactive Streamlit application where users can enter journey details, view predicted durations, and explore model performance through visualizations.

🔗 **Live Demo:** [Try RailWise AI](https://ragasandhiya05-railwise-ai-app-0ek8l1.streamlit.app/)

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Project Objectives](#project-objectives)
* [Project Levels](#project-levels)
* [Dataset Description](#dataset-description)
* [Technologies Used](#technologies-used)
* [Project Structure](#project-structure)
* [Installation and Setup](#installation-and-setup)
* [Running the Project](#running-the-project)
* [Model Comparison](#model-comparison)
* [Streamlit Application](#streamlit-application)
* [Visualizations](#visualizations)
* [Key Learnings](#key-learnings)
* [Limitations](#limitations)
* [Future Enhancements](#future-enhancements)
* [Author](#author)

---

## 📖 Project Overview

RailWise AI uses machine learning techniques to estimate scheduled train journey duration based on numerical journey details.

The project follows a structured workflow, beginning with dataset understanding and preprocessing, followed by exploratory data analysis, model training, model evaluation, and an interactive prediction application.

The trained models explore the relationship between:

* Total journey distance
* Number of intermediate stops
* Scheduled journey duration

The project compares different regression algorithms and integrates a selected model into a Streamlit application for interactive predictions.

---

## 🎯 Project Objectives

* Understand and analyze train journey data.
* Identify missing values, duplicate records, and potential data-quality issues.
* Clean the dataset and prepare relevant features.
* Explore relationships between distance, stops, and journey duration.
* Train and evaluate regression models.
* Compare model performance using MAE, RMSE, and R².
* Develop an interactive application for journey duration prediction.
* Present model results through visualizations and tables.

---

## 🧩 Project Levels

### Level 1 — Dataset Understanding

The first level focuses on exploring the dataset and understanding its structure.

**Tasks performed:**

* Inspect dataset dimensions and column names.
* Examine sample records and descriptive statistics.
* Analyze train-wise origin and destination information.
* Explore journey distance, number of stations, and intermediate stops.
* Check missing values and exact duplicate records.
* Generate dataset overview and route-related summaries.

**Outputs:**

* Dataset overview
* Train-wise route information
* Distance and stop statistics
* Data-quality report

### Level 2 — Data Cleaning and Feature Engineering

This level prepares the dataset for machine learning.

**Tasks performed:**

* Handle missing and duplicate records.
* Validate required numerical features.
* Prepare departure and arrival time information.
* Derive scheduled journey duration.
* Handle journeys that cross midnight.
* Create relevant numerical features for model training.

**Key features:**

* `Total_Distance`
* `Number_of_Stops`
* `Journey_Duration_Minutes`
* `Journey_Duration_Hours`

### Level 3 — Exploratory Data Analysis and Visualization

This level explores patterns and relationships within the cleaned dataset.

**Visualizations and analysis:**

* Distance vs. journey duration
* Intermediate stops vs. journey duration
* Numerical correlation matrix
* Correlation heatmap
* Pivot table summarizing journey duration by number of stops

These visualizations help examine how journey characteristics relate to scheduled travel time.

### Level 4 — Model Training and Evaluation

A Multiple Linear Regression model is trained using the selected numerical features.

**Workflow:**

1. Separate input features and target variable.
2. Split the dataset into training and testing sets.
3. Train the Linear Regression model.
4. Generate predictions on the test set.
5. Evaluate performance using MAE and RMSE.
6. Visualize actual vs. predicted journey durations.

**Input features:**

* Total distance
* Number of intermediate stops

**Target variable:** `Journey_Duration_Minutes`

### Level 5 — Model Comparison

Multiple regression models are trained and evaluated using the same train-test split.

**Models compared:**

| Model                      | Input Features                        |
| -------------------------- | ------------------------------------- |
| Basic Linear Regression    | Total distance                        |
| Multiple Linear Regression | Total distance and intermediate stops |
| Random Forest Regressor    | Total distance and intermediate stops |

**Evaluation metrics:**

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The comparison results are saved for analysis. The model with the lowest test-set MAE is selected and saved for use in the prediction application.

### Level 6 — Interactive Prediction System

The final level provides an interactive Streamlit application.

**Application features:**

* Enter total journey distance in kilometers.
* Enter the number of intermediate stops.
* Predict scheduled journey duration.
* View the predicted duration in minutes and hours.
* Explore model comparison results.
* View actual vs. predicted test results.

The application loads the saved model and uses the selected numerical inputs to generate predictions.

---

## 📊 Dataset Description

The project uses `Dataset1.csv` as its original train journey dataset.

The data is processed into a train-wise dataset containing journey-level information.

**Key columns in the processed dataset:**

| Column                     | Description                                |
| -------------------------- | ------------------------------------------ |
| `Train_No`                 | Train number                               |
| `Origin_Code`              | Origin station code                        |
| `Origin_Station`           | Origin station name                        |
| `Destination_Code`         | Destination station code                   |
| `Destination_Station`      | Destination station name                   |
| `Total_Distance`           | Total journey distance in kilometers       |
| `Total_Stations`           | Total stations associated with the journey |
| `Number_of_Stops`          | Number of intermediate stops               |
| `Departure_Minutes`        | Departure time represented in minutes      |
| `Arrival_Minutes`          | Arrival time represented in minutes        |
| `Journey_Duration_Minutes` | Scheduled journey duration in minutes      |
| `Journey_Duration_Hours`   | Scheduled journey duration in hours        |

**Processed dataset size:** 11,113 train-level records and 12 columns.

The cleaned dataset is saved as:

`data/cleaned_trains.csv`

---

## 🛠️ Technologies Used

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Core programming language                |
| Pandas           | Data manipulation and analysis           |
| NumPy            | Numerical operations                     |
| Matplotlib       | Data visualization                       |
| Seaborn          | Statistical visualizations               |
| Scikit-learn     | Machine learning and evaluation          |
| Joblib           | Saving and loading trained models        |
| Streamlit        | Interactive web application              |
| Jupyter Notebook | Exploratory analysis and experimentation |

---

## 📁 Project Structure

```text
RailWise-AI/
│
├── data/
│   ├── cleaned_trains.csv
│   └── quality_report.csv
│
├── models/
│   ├── best_model.pkl
│   └── linear_regression_model.pkl
│
├── notebooks/
│   └── RailWise_AI_Levels_1_to_5.ipynb
│
├── outputs/
│   ├── level1_dataset_overview.csv
│   ├── level1_distance_stop_statistics.csv
│   ├── level1_quality_checks.csv
│   ├── level1_train_wise_routes.csv
│   ├── level3_correlation_heatmap.png
│   ├── level3_correlation_matrix.csv
│   ├── level3_distance_vs_duration.png
│   ├── level3_stops_pivot_table.csv
│   ├── level3_stops_vs_duration.png
│   ├── level4_actual_vs_predicted.png
│   ├── level4_model_metrics.csv
│   ├── level4_predictions.csv
│   ├── level5_model_comparison.csv
│   ├── level5_predictions.csv
│   └── ...
│
├── app.py
├── prepare_data.py
├── eda_analysis.py
├── train_model.py
├── compare_models.py
├── requirements.txt
└── README.md
```

*Note: The `outputs/` folder may contain additional generated charts and CSV files.*

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/RagaSandhiya05/RailWise-AI.git
```

### 2. Navigate to the project folder

```bash
cd RailWise-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the following commands from the project root directory.

### Step 1 — Prepare the dataset

```bash
python prepare_data.py
```

This processes the original dataset and generates the cleaned train-level dataset and quality report.

Ensure that `Dataset1.csv` is available at the location expected by the script.

### Step 2 — Run dataset analysis

```bash
python eda_analysis.py
```

This generates dataset summaries, route statistics, and exploratory visualizations.

### Step 3 — Train the Linear Regression model

```bash
python train_model.py
```

This trains and evaluates the Level 4 model.

### Step 4 — Compare machine learning models

```bash
python compare_models.py
```

This compares the regression models, saves evaluation results, and generates the selected model.

### Step 5 — Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 📈 Model Comparison

The following results were obtained from the project's Level 5 model comparison on the configured test split.

| Model                      | MAE (minutes) | RMSE (minutes) |     R² |
| -------------------------- | ------------: | -------------: | -----: |
| Random Forest              |       35.0940 |       150.4686 | 0.9501 |
| Multiple Linear Regression |       58.5568 |       157.4458 | 0.9454 |
| Basic Linear Regression    |       67.4139 |       167.3811 | 0.9383 |

### Understanding the metrics

**MAE — Mean Absolute Error**

Measures the average absolute difference between actual and predicted journey durations. Lower values indicate smaller average errors.

**RMSE — Root Mean Squared Error**

Measures prediction error while penalizing larger errors more strongly. Lower values indicate smaller errors.

**R² Score**

Measures the proportion of variation in the target explained by the model on the evaluated test split. It is not the percentage of predictions that are correct.

The Random Forest model had the lowest MAE and RMSE and the highest R² among the three models in this comparison.

---

## 💻 Streamlit Application

RailWise AI includes an interactive application for estimating scheduled train journey duration.

### User Inputs

* Total journey distance (km)
* Number of intermediate stops

### Prediction Output

* Estimated journey duration in minutes
* Estimated journey duration in hours
* Model comparison table
* Actual vs. predicted visualization
* Sample test predictions

The application loads the saved model from the `models/` directory.

---

## 📷 Visualizations

The project generates visualizations to support dataset exploration and model evaluation.

### Level 3 — Exploratory Data Analysis

* Distance vs. journey duration
* Stops vs. journey duration
* Correlation heatmap
* Pivot table of journey duration by number of stops

### Level 4 — Model Evaluation

* Actual vs. predicted journey duration

### Level 5 — Model Comparison

* MAE comparison chart
* RMSE comparison chart
* Actual vs. predicted results for the selected model

To display screenshots in this README, create a `screenshots/` folder in the repository and add your captured images. You can then embed them using relative paths, for example:

```markdown
![RailWise AI Application](screenshots/railwise-app.png)
```

---

## 💡 Key Learnings

* Dataset inspection and preprocessing
* Feature engineering for regression problems
* Exploratory data analysis and visualization
* Training and evaluating regression models
* Comparing machine learning performance metrics
* Saving and loading trained models with Joblib
* Building an interactive ML application using Streamlit
* Presenting model results through visualizations

---

## ⚠️ Limitations

* The model estimates scheduled journey duration; it does not predict live train delays or real-time running times.
* The current prediction inputs are total distance and number of intermediate stops. Origin and destination names are not directly used as model features.
* Prediction quality depends on the quality and representativeness of the dataset.
* Unusually long journeys and large prediction errors should be reviewed for possible data-quality issues.
* The reported model metrics are based on the project's configured test split. They should not be interpreted as a guarantee of future prediction accuracy.

---

## 🚀 Future Enhancements

* Incorporate origin and destination station information into the model.
* Explore additional train and route features.
* Improve handling of unusual journey durations and outliers.
* Evaluate models using cross-validation and a separate final test set.
* Enhance the interactive application with train and station selection.
* Explore additional algorithms and feature engineering techniques.
* Deploy the application for public access.

---

## 👩‍💻 Author

**Raga Sandhiya R**

---

⭐ If you find this project interesting, feel free to explore the repository and its implementation.
