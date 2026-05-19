import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import validate_input_columns   # now it will work

def test_validate_input_columns_success(sample_raw_data):
    required = ['employee_id', 'employee_seniority_years']
    # Should not raise
    validate_input_columns(sample_raw_data, required)

def test_validate_input_columns_missing(sample_raw_data):
    required = ['missing_col', 'another_missing']
    with pytest.raises(ValueError, match="Missing required columns: missing_col, another_missing"):
        validate_input_columns(sample_raw_data, required)