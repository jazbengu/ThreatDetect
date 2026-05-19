# ThreatDetect

ThreatDetect is a Streamlit-based insider threat detection prototype that analyzes employee activity data and flags potentially risky behavior. It uses a trained XGBoost classifier together with an Isolation Forest anomaly detector to produce organisation-level risk summaries and explainable employee-level insights.

## Demo

The app can be run locally using Streamlit. A deployed demo was previously published at:

- https://threatdetectcos720.streamlit.app/

## Key Features

- Batch risk detection from an uploaded CSV file
- Organisational threat summary with counts, probability distribution, and risk tiers
- Per-employee explainability using SHAP values
- Exploratory Data Analysis (EDA) for uploaded or sample datasets
- CSV export of prediction results
- Visualizations for feature importance, risk distribution, and anomaly detection

## Repository Structure

- `streamlit_app.py` — main Streamlit application
- `requirements.txt` — Python package dependencies
- `AI_Model_Code/insider_threat_model.pkl` — saved inference pipeline and trained model artifacts
- `AI_Model_Code/insider_threat_clean_dataset.csv` — bundled sample dataset for EDA
- `AI_Model_Code/cos720_ai_model_FINAL.ipynb` — model development notebook
- `assets/app_styling.css` — custom UI styling for the Streamlit app
- `unit testing/` — test files for feature preparation and explanation logic
- `docs/TECHNICAL_DOCUMENTATION.md` — technical documentation for the project

## Software Requirements

- Python 3.8 or later
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `plotly`
- `xgboost`
- `shap`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jazbengu/ThreatDetect.git
cd ThreatDetect
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the app with:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Using ThreatDetect

### Organisational Search via CSV

- Upload a CSV file containing employee and activity features.
- Click `Run Threat Detection` to calculate risk probabilities for each record.
- The app shows a summary of overall malicious vs normal predictions, risk probability distribution, and feature importance.
- You can expand the detailed results table and download the output as a CSV.
- Use the per-record explanation section to inspect why a specific employee was flagged.

### Exploratory Data Analysis

- Upload your own CSV or use the bundled sample dataset in `AI_Model_Code/insider_threat_clean_dataset.csv`.
- The page displays dataset shape, descriptive statistics, missing value information, and feature plots.
- Numeric columns can be explored with histograms, boxplots, scatter relationships, and correlation heatmaps.
- A quick Isolation Forest anomaly scan can highlight unusual records.

### Sidebar Pages

- `Organisational Search via CSV` is the main batch threat detection workflow.
- `Single Search` is shown in the sidebar but is not currently implemented in the app logic.
- `Exploratory Data Analysis` is available for dataset exploration and quick anomaly checks.

## Model Artifacts

`AI_Model_Code/insider_threat_model.pkl` contains a serialized model package with:

- `xgb_model`: trained XGBoost classifier
- `iso_forest`: trained Isolation Forest anomaly detector
- `scaler`: StandardScaler for numeric inputs
- `label_encoders`: encoders for categorical features
- `feature_columns`: ordered feature list
- `cat_cols`, `num_cols`, `bin_cols`: feature groupings
- `best_threshold`: classification threshold used for risk decisions
- `shap_explainer`: SHAP explainer for explainability

The model file is loaded by `streamlit_app.py` and used to preprocess inputs, compute risk probabilities, and generate explanations.

## Testing

The repository includes tests under `unit testing/`.

Run the test suite with:

```bash
pytest "unit testing/"
```

## Notes and Troubleshooting

- If Streamlit fails to start, make sure your virtual environment is active and dependencies are installed.
- If the app reports missing required columns, verify that your uploaded CSV includes the expected employee and activity fields.
- The model file was serialized with older scikit-learn versions, so loading may generate compatibility warnings if your environment uses newer versions.
- If the built-in sample dataset is not available, upload your own CSV to continue using the EDA page.

## Contribution

This repository is configured for enhancement and testing. Future improvements may include expanding feature coverage, and retraining the model with additional insider threat data.
