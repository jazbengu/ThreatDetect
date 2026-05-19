import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import load_model, prepare_features  # adjust

def test_end_to_end_prediction(mock_model_package, sample_raw_data, mocker):
    # Mock load_model to return our mock package
    mocker.patch('streamlit_app.load_model', return_value=mock_model_package)
    
    model = load_model()  # returns mock
    df, x_append, iso_scores = prepare_features(sample_raw_data, model)
    
    xgb_model = model['xgb_model']
    probs = xgb_model.predict_proba(x_append)[:, 1]
    preds = (probs >= model['best_threshold']).astype(int)
    
    assert len(preds) == len(sample_raw_data)
    # Check that at least one prediction is made (no crash)
    assert set(preds).issubset({0,1})