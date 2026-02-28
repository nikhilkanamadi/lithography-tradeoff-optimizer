"""Tests for the SyntheticSimulator."""

import pytest

from lto.schemas import JobParameters, PatternComplexity
from lto.simulator.synthetic import SyntheticSimulator


@pytest.fixture
def simulator():
    return SyntheticSimulator(noise_scale=0.01, seed=42)


@pytest.fixture
def default_params():
    return JobParameters(na=0.33, dose_mj_cm2=15.0)


class TestSyntheticSimulator:

    def test_run_job_returns_result(self, simulator, default_params):
        result = simulator.run_job(default_params)
        assert result.job_id == default_params.job_id
        assert result.adapter_type == "synthetic"
        assert result.simulator_version.startswith("synthetic")

    def test_outputs_in_valid_range(self, simulator, default_params):
        result = simulator.run_job(default_params)
        assert result.outputs.resolution_nm > 0
        assert result.outputs.depth_of_focus_nm > 0
        assert 0 <= result.outputs.pattern_fidelity <= 1
        assert result.outputs.compute_time_s > 0
        assert 0 <= result.outputs.accuracy_vs_physics <= 1
        assert 0 <= result.outputs.yield_prediction <= 1

    def test_tradeoff_signals_in_range(self, simulator, default_params):
        result = simulator.run_job(default_params)
        signals = result.tradeoff_signals
        assert 0 <= signals.speed_vs_accuracy <= 1
        assert 0 <= signals.resolution_vs_dof <= 1
        assert 0 <= signals.cost_vs_fidelity <= 1
        assert 0 <= signals.surrogate_reliability <= 1
        assert 0 <= signals.yield_risk <= 1
        assert 0 <= signals.overall_health <= 1

    def test_physics_sim_slower_than_surrogate(self, simulator):
        """Full physics simulation should take more compute than surrogate."""
        params_surrogate = JobParameters(na=0.33, dose_mj_cm2=15.0, use_ai_surrogate=True, grid_size_nm=1.0)
        params_physics = JobParameters(na=0.33, dose_mj_cm2=15.0, use_ai_surrogate=False, grid_size_nm=1.0)
        r_surr = simulator.run_job(params_surrogate)
        r_phys = simulator.run_job(params_physics)
        assert r_phys.outputs.compute_time_s > r_surr.outputs.compute_time_s

    def test_physics_sim_perfect_accuracy(self, simulator):
        """Full physics simulation should have accuracy == 1.0."""
        params = JobParameters(na=0.33, dose_mj_cm2=15.0, use_ai_surrogate=False)
        result = simulator.run_job(params)
        assert result.outputs.accuracy_vs_physics == 1.0

    def test_generate_batch(self, simulator):
        results = simulator.generate_batch(10)
        assert len(results) == 10
        for r in results:
            assert r.adapter_type == "synthetic"

    def test_generate_dataframe(self, simulator):
        df = simulator.generate_dataframe(50)
        assert len(df) == 50
        assert "speed_vs_accuracy" in df.columns
        assert "yield_risk" in df.columns
        assert "na" in df.columns

    def test_deterministic_with_seed(self):
        sim1 = SyntheticSimulator(seed=123)
        sim2 = SyntheticSimulator(seed=123)
        params = JobParameters(na=0.33, dose_mj_cm2=15.0)
        r1 = sim1.run_job(params)
        r2 = sim2.run_job(params)
        assert r1.outputs.resolution_nm == r2.outputs.resolution_nm
        assert r1.tradeoff_signals.overall_health == r2.tradeoff_signals.overall_health

    def test_health_check(self, simulator):
        health = simulator.health_check()
        assert health.healthy is True
        assert "synthetic" in health.message.lower()
