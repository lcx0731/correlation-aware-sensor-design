"""Linear-Gaussian sensor design; information is measured in nats."""

from .core import GaussianDesign, Selection, exact_select, greedy_select

__all__ = ["GaussianDesign", "Selection", "exact_select", "greedy_select"]
