"""Optical physics models for lithography simulation.

These implement the foundational equations of optical lithography:
- Rayleigh resolution criterion
- Depth of focus (Rayleigh criterion)
- Modulation Transfer Function (MTF)
- Aerial image contrast estimation
"""

from __future__ import annotations

import math


def resolution(wavelength_nm: float, na: float, k1: float = 0.25) -> float:
    """Compute minimum resolvable feature size using Rayleigh criterion.

    Resolution = k1 * λ / NA

    Args:
        wavelength_nm: Illumination wavelength in nanometers.
        na: Numerical aperture of the projection lens.
        k1: Process factor (0.25 = theoretical limit, typical 0.3–0.8).

    Returns:
        Minimum resolvable half-pitch in nanometers.
    """
    return k1 * wavelength_nm / na


def depth_of_focus(wavelength_nm: float, na: float, k2: float = 0.5) -> float:
    """Compute depth of focus using Rayleigh criterion.

    DoF = k2 * λ / NA²

    Higher NA improves resolution but dramatically reduces DoF —
    this is the fundamental Resolution vs DoF tradeoff.

    Args:
        wavelength_nm: Illumination wavelength in nm.
        na: Numerical aperture.
        k2: Focus factor (0.5 = classical Rayleigh).

    Returns:
        Depth of focus in nanometers.
    """
    return k2 * wavelength_nm / (na ** 2)


def mtf(spatial_frequency: float, na: float, wavelength_nm: float) -> float:
    """Compute the Modulation Transfer Function for coherent imaging.

    The MTF describes how well the optical system transfers contrast
    at a given spatial frequency. When MTF → 0, the pattern is unresolvable.

    Args:
        spatial_frequency: Spatial frequency in 1/nm.
        na: Numerical aperture.
        wavelength_nm: Wavelength in nm.

    Returns:
        MTF value between 0.0 and 1.0.
    """
    cutoff = 2 * na / wavelength_nm
    if spatial_frequency >= cutoff:
        return 0.0

    normalized = spatial_frequency / cutoff
    return (2 / math.pi) * (
        math.acos(normalized)
        - normalized * math.sqrt(1 - normalized ** 2)
    )


def aerial_image_contrast(na: float, sigma: float, wavelength_nm: float, pitch_nm: float) -> float:
    """Estimate aerial image contrast (simplified model).

    Contrast depends on the coherence setting (sigma) and the feature pitch
    relative to the resolution limit.

    Args:
        na: Numerical aperture.
        sigma: Partial coherence factor (0=coherent, 1=incoherent).
        wavelength_nm: Wavelength in nm.
        pitch_nm: Feature pitch in nm.

    Returns:
        Estimated contrast between 0.0 and 1.0.
    """
    spatial_freq = 1.0 / pitch_nm
    base_mtf = mtf(spatial_freq, na, wavelength_nm)

    # Partial coherence reduces contrast for dense features
    coherence_penalty = 1.0 - 0.3 * sigma
    return max(0.0, min(1.0, base_mtf * coherence_penalty))


def process_window_area(
    dose_latitude_pct: float,
    dof_nm: float,
) -> float:
    """Compute process window area — a key manufacturing margin metric.

    Larger process window = more manufacturing latitude = higher yield.

    Args:
        dose_latitude_pct: Exposure latitude in percent.
        dof_nm: Depth of focus in nm.

    Returns:
        Process window area (arbitrary units, proportional to margin).
    """
    return dose_latitude_pct * dof_nm
