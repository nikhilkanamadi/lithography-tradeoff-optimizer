"""Simulator package — synthetic and adapter-based data generation."""

from lto.simulator.base import SimulatorInterface
from lto.simulator.synthetic import SyntheticSimulator

__all__ = ["SimulatorInterface", "SyntheticSimulator"]
