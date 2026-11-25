"""Configurable still-image industrial inspection platform."""

from .config import ProductConfig, load_product_config
from .policy import DecisionPolicy, OperationalDecision

__all__ = ["DecisionPolicy", "OperationalDecision", "ProductConfig", "load_product_config"]
