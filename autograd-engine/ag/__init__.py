"""A module for automatic differentiation of scalar functions."""
import math
from typing import Any

from ag.scalar import Parameter, Scalar

__all__ = ["Parameter", "Scalar", "max"]
LOG_EPSILON: float = 1e-12


def isclose(x: Any, y: Any, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Return True if the values x and y are close to each other and False otherwise.

    This is a wrapper around math.isclose() that also works for ag.Scalar objects.
    """
    if isinstance(x, Scalar):
        x_float: float = x.data
    else:
        x_float = x
    if isinstance(y, Scalar):
        y_float: float = y.data
    else:
        y_float = y
    return math.isclose(x_float, y_float, rel_tol=rel_tol, abs_tol=abs_tol)


def max(x: Scalar, y: Scalar) -> Scalar:
    """Compute the maximum of two scalars."""
    return x.max(y)


def min(x: Scalar, y: Scalar) -> Scalar:
    """Compute the minimum of two scalars."""
    return x.min(y)


def clip(x: Scalar, min_value: float, max_value: float) -> Scalar:
    """Clip the value of a scalar between a minimum and a maximum value."""
    return x.clip(min_value, max_value)


def sigmoid(x: Scalar) -> Scalar:
    """Compute the sigmoid of a scalar."""
    return x.sigmoid()


def relu(x: Scalar) -> Scalar:
    """Compute the rectified linear unit of a scalar."""
    return x.relu()


def tanh(x: Scalar) -> Scalar:
    """Compute the hyperbolic tangent of a scalar."""
    return x.tanh()


def exp(x: Scalar) -> Scalar:
    """Compute the exponential of a scalar."""
    return x.exp()


def log(x: Scalar, safe: bool = False) -> Scalar:
    """Compute the natural logarithm of a scalar."""
    if safe:
        return x.max(LOG_EPSILON).log()
    return x.log()
