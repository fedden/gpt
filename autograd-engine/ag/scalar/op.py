"""Functions for scalars."""
from typing import List, Union

from ag.scalar import Parameter, Scalar


def sum(*args: Union[List[Scalar], Scalar]) -> Scalar:
    """Compute the sum of a variable number of scalars."""
    if len(args) == 0:
        raise ValueError("sum() requires at least one argument")
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    total: Scalar = Scalar(0.0)
    for s in args:
        total += s
    return total
