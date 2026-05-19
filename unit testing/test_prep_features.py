import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import prepare_features   # now it will work # adjust import

def test_prepare_features_normal_case(sample_raw_data, mock_model_package):
    df, x_append, iso_scores = prepare_features(sample_raw_data, mock_model_package)
    
    # Check output shapes
    assert x_append.shape[0] == len(sample_raw_data)
    # Number of feature columns should be len(mock_model_package['feature_columns'])
    expected_feat_count = len(mock_model_package['feature_columns'])
    assert x_append.shape[1] == expected_feat_count
    
    # iso_scores should be 1D array of same length
    assert iso_scores.shape == (len(sample_raw_data),)
    
    # Check engineered columns exist in returned df (optional, but they should be added)
    assert 'print_ratio' in df.columns
    assert 'file_ratio' in df.columns

def test_prepare_features_unseen_category(sample_raw_data, mock_model_package):
    # Introduce a new campus that encoder never saw
    sample_raw_data.loc[0, 'employee_campus'] = 'Z'
    with pytest.raises(ValueError, match="contains unseen categories"):
        prepare_features(sample_raw_data, mock_model_package)

def test_prepare_features_handles_inf_ratios(sample_raw_data, mock_model_package):
    # Force division by zero to create inf
    sample_raw_data['num_printed_pages_off_hours'] = 0
    df, x_append, iso_scores = prepare_features(sample_raw_data, mock_model_package)
    # Should not crash, and inf should be replaced
    assert np.isfinite(x_append).all()