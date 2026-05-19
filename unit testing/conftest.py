import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap

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
def mock_model_package(sample_raw_data):
    # Define columns exactly as in the real model
    cat_cols = ['employee_campus']
    bin_cols = ['has_criminal_record', 'is_contractor', 'has_foreign_citizenship',
                'entry_during_weekend', 'late_exit_flag']
    num_cols = ['employee_seniority_years', 'total_printed_pages', 'num_printed_pages_off_hours',
                'total_files_burned', 'trip_day_number', 'num_entries', 'num_unique_campus']
    
    # LabelEncoder for categorical
    le = LabelEncoder()
    le.fit(sample_raw_data['employee_campus'])
    
    # StandardScaler for numerical (fit on raw numeric columns)
    scaler = StandardScaler()
    scaler.fit(sample_raw_data[num_cols].fillna(0))
    
    # --- Build the full feature set that prepare_features will produce ---
    # 1. Raw numerical (already in scaler)
    # 2. Encoded categorical
    # 3. Engineered ratios
    engineered = ['print_ratio', 'file_ratio', 'risk_ratio', 'access_ratio', 'afterhrs_ratio']
    # 4. IsolationForest anomaly score (added later)
    feature_columns = num_cols + cat_cols + engineered + ['isolation_forest_anomaly_score']
    n_features = len(feature_columns)  # e.g., 7+1+5+1 = 14
    
    # Instead of training on raw num_cols (7), train on the full feature_cols (excluding iso score)
    full_features_for_iso = num_cols + cat_cols + engineered   # length = 7+1+5 = 13
    iso_forest = IsolationForest(random_state=42)
    # Use sample_raw_data with those columns engineered (create a dummy engineered dataset)
    dummy_engineered = sample_raw_data.copy()
    # Add dummy engineered columns (since they are not in raw data)
    for col in engineered:
        dummy_engineered[col] = np.random.rand(len(sample_raw_data))
    X_iso = dummy_engineered[full_features_for_iso]
    iso_forest.fit(X_iso)
    # --- Train XGBoost on dummy data with the correct number of features (14) ---
    # We'll create random data of shape (100, n_features) and binary labels
    np.random.seed(42)
    dummy_X = np.random.rand(100, n_features)
    dummy_y = np.random.randint(0, 2, 100)
    xgb_model = xgb.XGBClassifier(n_estimators=2, random_state=42)
    xgb_model.fit(dummy_X, dummy_y)
    
    # SHAP explainer for XGBoost (works on the same n_features)
    explainer = shap.TreeExplainer(xgb_model)
    
    model_package = {
        'xgb_model': xgb_model,
        'iso_forest': iso_forest,
        'scaler': scaler,
        'label_encoders': {'employee_campus': le},
        'cat_cols': cat_cols,
        'bin_cols': bin_cols,
        'num_cols': num_cols,
        'feature_columns': feature_columns,
        'best_threshold': 0.5,
        'shap_explainer': explainer
    }
    return model_package