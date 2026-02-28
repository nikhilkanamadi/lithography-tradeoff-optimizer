"""Tests for the ML tradeoff pipeline: features → XGBoost → TradeoffModel."""

import pytest
import numpy as np

from lto.schemas import JobParameters
from lto.simulator.synthetic import SyntheticSimulator
from lto.ml.features.engineering import extract_features, get_feature_columns
from lto.ml.engine import TradeoffModel


@pytest.fixture(scope="module")
def trained_model():
    """Train a model once for all tests in this module."""
    model = TradeoffModel(model_dir="/tmp/lto_test_models")
    model.train_from_simulator(n_samples=500, seed=42)
    return model


@pytest.fixture
def test_params():
    return JobParameters(na=0.33, dose_mj_cm2=15.0)


class TestFeatureEngineering:

    def test_extract_features_shape(self):
        sim = SyntheticSimulator(seed=42)
        df = sim.generate_dataframe(100)
        features = extract_features(df)
        assert len(features) == 100
        assert len(features.columns) == len(get_feature_columns())

    def test_features_no_nulls(self):
        sim = SyntheticSimulator(seed=42)
        df = sim.generate_dataframe(50)
        features = extract_features(df)
        assert not features.isnull().any().any()


class TestTradeoffModel:

    def test_model_trains_successfully(self, trained_model):
        assert trained_model.is_trained

    def test_prediction_returns_all_scores(self, trained_model, test_params):
        prediction = trained_model.predict(test_params)
        expected_keys = {
            "speed_vs_accuracy",
            "resolution_vs_dof",
            "cost_vs_fidelity",
            "surrogate_reliability",
            "yield_risk",
        }
        assert set(prediction.predictions.keys()) == expected_keys

    def test_scores_in_valid_range(self, trained_model, test_params):
        prediction = trained_model.predict(test_params)
        for name, score_ci in prediction.predictions.items():
            assert 0.0 <= score_ci.score <= 1.0, f"{name} score out of range"
            assert score_ci.ci_low <= score_ci.score <= score_ci.ci_high

    def test_prediction_has_uncertainty(self, trained_model, test_params):
        prediction = trained_model.predict(test_params)
        assert prediction.uncertainty is not None
        assert prediction.uncertainty.confidence_level is not None

    def test_prediction_has_constraints(self, trained_model, test_params):
        prediction = trained_model.predict(test_params)
        assert prediction.constraints is not None

    def test_inference_time_reasonable(self, trained_model, test_params):
        prediction = trained_model.predict(test_params)
        assert prediction.inference_time_ms < 1000  # Should be < 1 second

    def test_model_save_and_load(self, trained_model, test_params):
        # Save
        path = trained_model.save("test_model")

        # Load
        loaded = TradeoffModel.load(path)
        assert loaded.is_trained

        # Predictions should match
        pred1 = trained_model.predict(test_params)
        pred2 = loaded.predict(test_params)
        for name in pred1.predictions:
            assert abs(pred1.predictions[name].score - pred2.predictions[name].score) < 1e-6
