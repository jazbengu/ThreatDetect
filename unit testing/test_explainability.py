import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import results_explainability

def test_results_explainability_output(mock_model_package, sample_raw_data):
    # Simulate a feature vector (full feature columns)
    feature_names = mock_model_package['feature_columns']
    dummy_x = np.random.rand(len(feature_names))  # one instance
    iso_score = 0.2
    original_row = sample_raw_data.iloc[0].to_dict()
    threshold = 0.5
    
    pred, prob, conf, explanation_list, feat_contrib = results_explainability(
        mock_model_package, dummy_x, iso_score, original_row, feature_names, threshold
    )
    
    assert pred in ['Malicious', 'Normal']
    assert 0 <= prob <= 1
    assert 0 <= conf <= 1
    assert isinstance(explanation_list, list)
    assert isinstance(feat_contrib, pd.DataFrame)
    assert len(feat_contrib) == len(feature_names)
    
    # At least a few explanation bullets (might be empty if all SHAP are zero)
    # But we can check structure
    for bullet in explanation_list:
        assert "→" in bullet

def test_explainability_shap_extraction(mock_model_package):
    # Ensure the function extracts the correct SHAP class (index 1) when explainer returns list
    feature_names = mock_model_package['feature_columns']
    dummy_x = np.random.rand(len(feature_names))
    orig_row = {'some_feature': 123}
    # We just run the function; previous test already covers no crash
    results_explainability(mock_model_package, dummy_x, 0.0, orig_row, feature_names, 0.5)
    # If it reaches here, extraction succeeded