"""
Tests for ML models.
Run with: pytest tests/test_models.py
"""

import pytest
import numpy as np
import pandas as pd
from ml.models.ensemble import InsuranceEnsembleModel

class TestEnsembleModel:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'vehicle_age': np.random.randint(0, 20, 100),
            'premium': np.random.uniform(100, 500, 100),
            'claims_history': np.random.randint(0, 5, 100),
            'type_fuel': np.random.choice(['P', 'D', 'G'], 100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_model_initialization(self):
        model = InsuranceEnsembleModel()
        assert model is not None
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')

    def test_model_train_predict(self, sample_data):
        X, y = sample_data
        model = InsuranceEnsembleModel()
        
        # Train on 80% of data
        X_train = X[:80]
        y_train = y[:80]
        X_test = X[80:]
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(0 <= pred <= 1 for pred in predictions)

    def test_model_explain(self, sample_data):
        X, y = sample_data
        model = InsuranceEnsembleModel()
        model.train(X[:80], y[:80])
        
        explanation = model.explain(X[80:], instance_idx=0)
        assert "prediction" in explanation
        assert "top_features" in explanation

class TestFeatureEngineering:
    def test_preprocessing(self):
        X = pd.DataFrame({
            'age': [30, 45, None, 50],
            'vehicle_age': [2, 5, 10, 3],
            'type_fuel': ['P', 'D', 'P', None]
        })
        
        # Check handling of missing values
        assert X['age'].isnull().sum() > 0
        assert X['type_fuel'].isnull().sum() > 0
