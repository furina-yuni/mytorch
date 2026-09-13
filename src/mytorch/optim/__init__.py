"""Optimization algorithms."""

from . import lr_scheduler
from .adam import Adam, Adamax, AdamW, NAdam, RAdam
from .adaptive import Adadelta, Adagrad, RMSprop
from .classical import ASGD, Rprop
from .lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from .modern import Adafactor, Lion
from .optimizer import Optimizer
from .sgd import SGD

__all__ = [
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Lion",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "ExponentialLR",
    "LRScheduler",
    "LinearLR",
    "MultiStepLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
    "lr_scheduler",
]
