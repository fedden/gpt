"""Random tensor module."""
import math
import random

import ag
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


def permutation(x: int | Tensor) -> Tensor:
    """Return a tensor with values drawn from a permutation of `x`."""
    if isinstance(x, int):
        x = ag.arange(start=0, stop=x, step=1)
    assert isinstance(x, Tensor)
    random_indices: list[int] = random.sample(range(x.shape[0]), x.shape[0])
    # TODO(leonfedden):
    #   Implement indexing using lists of ints or int tensors and use that
    #   here. For now, let's just use a list comprehension and concat.
    elements = [x[i] for i in random_indices]
    return ag.stack(elements, axis=0)
