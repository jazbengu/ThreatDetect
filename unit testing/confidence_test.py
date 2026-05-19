import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import prepare_features   # now it will work

#to run test pytest "unit testing/"

@pytest.fixture
def sample_raw_data():
    """Minimal valid input DataFrame with required columns."""
    return pd.DataFrame({
        'employee_id': [1, 2],
        'employee_seniority_years': [2, 10],
        'total_printed_pages': [100, 50],
        'num_printed_pages_off_hours': [10, 5],
        'total_files_burned': [0, 20],
        'has_criminal_record': [0, 1],
        'is_contractor': [1, 0],
        'has_foreign_citizenship': [0, 0],
        'entry_during_weekend': [0, 1],
        'late_exit_flag': [1, 0],
        'employee_campus': ['A', 'B'],
        'trip_day_number': [0, 3],
        'num_entries': [5, 2],
        'num_unique_campus': [1, 2]
    })

@pytest.fixture
def mock_model_package(tmp_path, sample_raw_data):
    """Create a lightweight model package for testing."""
    # Dummy encoders and scaler
    cat_cols = ['employee_campus']
    le = LabelEncoder()
    le.fit(sample_raw_data['employee_campus'])
    num_cols = ['employee_seniority_years', 'total_printed_pages', 'num_printed_pages_off_hours',
                'total_files_burned', 'trip_day_number', 'num_entries', 'num_unique_campus']
    scaler = StandardScaler()
    scaler.fit(sample_raw_data[num_cols].fillna(0))

    # Dummy IsolationForest
    iso = IsolationForest(random_state=42)
    iso.fit(sample_raw_data[num_cols].fillna(0))

    # Dummy XGBoost model (binary classification)
    xgb_model = xgb.XGBClassifier(n_estimators=2, random_state=42)
    # Create dummy features (including engineered + iso score)
    dummy_features = np.random.rand(2, 15)  # adjust size to match feature_columns
    dummy_labels = [0, 1]
    xgb_model.fit(dummy_features, dummy_labels)

    # SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    feature_columns = (num_cols + cat_cols +
                       ['print_ratio', 'file_ratio', 'risk_ratio', 'access_ratio', 'afterhrs_ratio',
                        'isolation_forest_anomaly_score'])

    model_package = {
        'xgb_model': xgb_model,
        'iso_forest': iso,
        'scaler': scaler,
        'label_encoders': {'employee_campus': le},
        'cat_cols': cat_cols,
        'bin_cols': ['has_criminal_record', 'is_contractor', 'has_foreign_citizenship',
                     'entry_during_weekend', 'late_exit_flag'],
        'num_cols': num_cols,
        'feature_columns': feature_columns,
        'best_threshold': 0.5,
        'shap_explainer': explainer
    }
    return model_package