"""A basic autograd library for learning purposes."""
from __future__ import annotations
from typing import Any, List

import numpy as np

from ag.tensor import Tensor


a = Tensor([[1, 2, 3]], requires_grad=True)
b = Tensor(np.zeros((3, 3)), requires_grad=True)
c = a + b
d = (b - 2) * (b + 10)
e = c @ d
breakpoint()
