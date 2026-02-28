"""Feature engineering for the ML tradeoff model.

Transforms raw JobParameters into a feature vector suitable for ML models.
Includes physics-derived features, interaction terms, and normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lto.simulator.models.optical import depth_of_focus, resolution


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

RAW_FEATURE_COLS = [
    "na",
    "wavelength_nm",
    "dose_mj_cm2",
    "sigma",
    "resist_thickness_nm",
    "grid_size_nm",
    "use_ai_surrogate",
    "job_class",
]

PHYSICS_FEATURE_COLS = [
    "resolution_nm_feat",
    "depth_of_focus_nm_feat",
    "na_squared",
    "dose_per_thickness",
]

INTERACTION_FEATURE_COLS = [
    "na_x_dose",
    "wavelength_over_na",
    "grid_x_complexity",
    "sigma_x_na",
]

TARGET_COLS = [
    "speed_vs_accuracy",
    "resolution_vs_dof",
    "cost_vs_fidelity",
    "surrogate_reliability",
    "yield_risk",
]

COMPLEXITY_MAP = {"simple": 0, "moderate": 1, "complex": 2, "extreme": 3}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all features from a simulation DataFrame.

    Takes raw parameter columns and produces physics + interaction features.

    Args:
        df: DataFrame with raw parameter columns from SyntheticSimulator.generate_dataframe()

    Returns:
        Feature DataFrame ready for model input.
    """
    features = pd.DataFrame(index=df.index)

    # Raw features
    features["na"] = df["na"]
    features["wavelength_nm"] = df["wavelength_nm"]
    features["dose_mj_cm2"] = df["dose_mj_cm2"]
    features["sigma"] = df["sigma"]
    features["resist_thickness_nm"] = df["resist_thickness_nm"]
    features["grid_size_nm"] = df["grid_size_nm"]
    features["use_ai_surrogate"] = df["use_ai_surrogate"].astype(float)
    features["job_class"] = df["job_class"].astype(float)

    # Physics-derived features
    features["resolution_nm_feat"] = df.apply(
        lambda r: resolution(r["wavelength_nm"], r["na"]), axis=1
    )
    features["depth_of_focus_nm_feat"] = df.apply(
        lambda r: depth_of_focus(r["wavelength_nm"], r["na"]), axis=1
    )
    features["na_squared"] = df["na"] ** 2
    features["dose_per_thickness"] = df["dose_mj_cm2"] / df["resist_thickness_nm"]

    # Interaction features
    features["na_x_dose"] = df["na"] * df["dose_mj_cm2"]
    features["wavelength_over_na"] = df["wavelength_nm"] / df["na"]
    features["sigma_x_na"] = df["sigma"] * df["na"]

    # Pattern complexity as numeric
    if "pattern_complexity" in df.columns:
        features["grid_x_complexity"] = df["grid_size_nm"] * df["pattern_complexity"].map(
            COMPLEXITY_MAP
        ).fillna(1)
    else:
        features["grid_x_complexity"] = df["grid_size_nm"]

    return features


def get_feature_columns() -> list[str]:
    """Return the ordered list of all feature columns."""
    return RAW_FEATURE_COLS + PHYSICS_FEATURE_COLS + INTERACTION_FEATURE_COLS


def get_target_columns() -> list[str]:
    """Return the ordered list of target columns."""
    return TARGET_COLS


class FeatureScaler:
    """Wrapper around StandardScaler that tracks column names."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns: list[str] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the scaler and transform features."""
        self.columns = list(df.columns)
        self._fitted = True
        return self.scaler.fit_transform(df.values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")
        return self.scaler.transform(df[self.columns].values)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
