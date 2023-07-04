"""A basic autograd library for learning purposes."""
from __future__ import annotations
from typing import Any, List

import ag
from ag import Parameter, Scalar
from ag.loss import mse, binary_cross_entropy
from ag.optimiser import SGDOptimiser


