"""Random tensor module."""
import math
import random

from ag.tensor import Tensor


def seed(seed: int) -> None:
    """Seed the random number generator."""
    random.seed(seed)


def uniform(
    shape: tuple[int, ...], low: float = 0.0, high: float = 1.0, **kwargs
) -> Tensor:
    """Return a tensor with values drawn from a uniform distribution.

    Args:
        shape: Shape of the tensor.
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.

    Returns:
        A tensor with values drawn from a uniform distribution.

    """
    return Tensor(
        data=[random.uniform(low, high) for _ in range(math.prod(shape))],
        shape=shape,
        **kwargs,
    )


def normal(
    shape: tuple[int, ...], mean: float = 0.0, std: float = 1.0, **kwargs
) -> Tensor:
    """Return a tensor with values drawn from a normal distribution.

    Args:
        shape: Shape of the tensor.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.

    Returns:
        A tensor with values drawn from a normal distribution.

    """
    return Tensor(
        data=[random.gauss(mean, std) for _ in range(math.prod(shape))],
        shape=shape,
        **kwargs,
    )
