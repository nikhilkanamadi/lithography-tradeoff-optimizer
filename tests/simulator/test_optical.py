"""Tests for optical physics models."""

import math
import pytest

from lto.simulator.models.optical import (
    aerial_image_contrast,
    depth_of_focus,
    mtf,
    process_window_area,
    resolution,
)


class TestResolution:
    """Test Rayleigh resolution criterion."""

    def test_euv_standard(self):
        """EUV (13.5nm) with NA=0.33 → typical resolution ~10nm."""
        res = resolution(13.5, 0.33, k1=0.25)
        assert 9.0 < res < 12.0

    def test_arf_standard(self):
        """ArF (193nm) with NA=0.55 → typical resolution ~88nm."""
        res = resolution(193.0, 0.55, k1=0.25)
        assert 80.0 < res < 100.0

    def test_higher_na_better_resolution(self):
        """Higher NA → smaller (better) resolution."""
        res_low_na = resolution(13.5, 0.25)
        res_high_na = resolution(13.5, 0.50)
        assert res_high_na < res_low_na

    def test_shorter_wavelength_better_resolution(self):
        """Shorter wavelength → better resolution."""
        res_euv = resolution(13.5, 0.33)
        res_arf = resolution(193.0, 0.33)
        assert res_euv < res_arf


class TestDepthOfFocus:
    """Test depth of focus calculation."""

    def test_euv_standard(self):
        """EUV with NA=0.33 → DoF should be reasonable."""
        dof = depth_of_focus(13.5, 0.33)
        assert dof > 0

    def test_higher_na_reduces_dof(self):
        """Higher NA dramatically reduces DoF — the core tradeoff."""
        dof_low = depth_of_focus(13.5, 0.25)
        dof_high = depth_of_focus(13.5, 0.50)
        assert dof_high < dof_low

    def test_dof_formula(self):
        """Verify exact formula: DoF = k2 * λ / NA²."""
        dof = depth_of_focus(13.5, 0.33, k2=0.5)
        expected = 0.5 * 13.5 / (0.33 ** 2)
        assert abs(dof - expected) < 1e-10

    def test_resolution_vs_dof_tradeoff(self):
        """Core tradeoff: improving resolution (high NA) degrades DoF."""
        # Low NA: worse resolution but better DoF
        res_low = resolution(13.5, 0.25)
        dof_low = depth_of_focus(13.5, 0.25)
        # High NA: better resolution but worse DoF
        res_high = resolution(13.5, 0.55)
        dof_high = depth_of_focus(13.5, 0.55)

        assert res_high < res_low   # Better resolution
        assert dof_high < dof_low   # But worse DoF


class TestMTF:
    """Test Modulation Transfer Function."""

    def test_zero_frequency(self):
        """MTF at zero frequency should be 1.0."""
        result = mtf(0.0, 0.33, 13.5)
        assert abs(result - 1.0) < 1e-6

    def test_cutoff_frequency(self):
        """MTF at cutoff frequency should be 0.0."""
        cutoff = 2 * 0.33 / 13.5
        result = mtf(cutoff, 0.33, 13.5)
        assert result == 0.0

    def test_above_cutoff(self):
        """MTF above cutoff should be 0.0."""
        cutoff = 2 * 0.33 / 13.5
        result = mtf(cutoff * 1.5, 0.33, 13.5)
        assert result == 0.0

    def test_monotonically_decreasing(self):
        """MTF should decrease with increasing spatial frequency."""
        freqs = [0.001, 0.01, 0.02, 0.03, 0.04]
        values = [mtf(f, 0.33, 13.5) for f in freqs]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


class TestProcessWindowArea:
    """Test process window area calculation."""

    def test_positive_area(self):
        area = process_window_area(10.0, 50.0)
        assert area == 500.0

    def test_zero_dose_latitude(self):
        area = process_window_area(0.0, 50.0)
        assert area == 0.0
