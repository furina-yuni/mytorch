"""Compatibility facade for additional dense optimizers."""

from .classical import ASGD, Rprop
from .modern import Adafactor, Lion

__all__ = ["ASGD", "Adafactor", "Lion", "Rprop"]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
