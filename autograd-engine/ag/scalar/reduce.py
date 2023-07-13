"""Functions for scalars."""
from typing import Callable, Union

from ag.scalar import Parameter, Scalar


def sum(*args: Union[list[Scalar], Scalar]) -> Scalar:
    """Compute the sum of a variable number of scalars."""
    return _reduce_helper(*args, fn=lambda x, y: x + y)


def prod(*args: Union[list[Scalar], Scalar]) -> Scalar:
    """Compute the product of a variable number of scalars."""
    return _reduce_helper(*args, fn=lambda x, y: x * y)


def max(*args: Union[list[Scalar], Scalar]) -> Scalar:
    """Compute the maximum of a variable number of scalars."""
    return _reduce_helper(*args, fn=lambda x, y: x.max(y))


def min(*args: Union[list[Scalar], Scalar]) -> Scalar:
    """Compute the minimum of a variable number of scalars."""
    return _reduce_helper(*args, fn=lambda x, y: x.min(y))


def _reduce_helper(
    *args: Union[list[Scalar], Scalar], fn: Callable[[Scalar, Scalar], Scalar]
) -> Scalar:
    """Reduce a variable number of scalars."""
    if len(args) == 0:
        raise ValueError("_reduce() requires at least one argument")
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    total: Scalar = Scalar(0.0)
    for s in args:
        total = fn(total, s)
    return total
