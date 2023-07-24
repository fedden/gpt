"""A module for automatic differentiation of scalar functions."""
import math
from typing import Any, Optional

from ag import ascii, loss, nn, optimiser, random, tensor, utils
from ag.scalar import Scalar
from ag.tensor import Tensor, Parameter

LOG_EPSILON: float = 1e-12

newaxis: Optional = None


class no_grad:
    """Context-manager that disabled gradient calculation."""

    def __init__(self) -> None:
        """Initialize a no_grad context manager."""
        self.prev: bool = False

    def __enter__(self) -> None:
        """Enter the no_grad context manager."""
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the no_grad context manager."""
        set_grad_enabled(self.prev)


def set_grad_enabled(enabled: bool) -> None:
    """Enable or disable gradient calculation."""
    Scalar._no_grad = not enabled
    Tensor._no_grad = not enabled


def is_grad_enabled() -> bool:
    """Return True if gradient calculation is enabled."""
    assert Scalar._no_grad == Tensor._no_grad, (
        f"Scalar._no_grad ({Scalar._no_grad}) and Tensor._no_grad "
        f"({Tensor._no_grad}) have diverged, this should never happen."
    )
    return not Scalar._no_grad


def stack(tensors: list, axis: int = 0) -> Tensor:
    """Stack tensors in an axis."""
    t = tensors[0]
    key = [slice(None) for _ in t.shape]
    key = key[:axis] + [newaxis] + key[axis:]
    return concatenate([t[key] for t in tensors], axis=axis)



def concatenate(tensors: list[Tensor], axis: Optional[int] = 0) -> Tensor:
    """Concatenate tensors along an axis."""
    assert len(tensors) > 0, "Must pass at least one tensor."
    if axis is None:
        # If axis is None, we concatenate along the first axis.
        axis = 0
        # We flatten the tensors into a single axis.
        tensors = [tensor.flatten() for tensor in tensors]
    # Check that all tensors have the same shape except for the axis which we
    # are concatenating along.
    shape = tensors[0].shape
    for tensor in tensors:
        assert not tensor.is_scalar, "Zero-dimensional arrays cannot be concatenated"
        assert (
            tensor.shape[:axis] == shape[:axis]
        ), f"Expected shape {shape[:axis]}, got {tensor.shape[:axis]}."
        assert (
            tensor.shape[axis + 1 :] == shape[axis + 1 :]
        ), f"Expected shape {shape[axis + 1 :]}, got {tensor.shape[axis + 1 :]}."
    # Concatenate the data in the right order as defined by the axis.
    new_shape = list(shape)
    new_shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
    # Use Python lists to concatenate the data in the right order.
    new_tensor: Tensor = zeros(new_shape)
    for i, tensor in enumerate(tensors):
        key = [slice(None)] * len(shape)
        key[axis] = slice(i * tensor.shape[axis], (i + 1) * tensor.shape[axis])
        new_tensor[key] = tensor
    return new_tensor


def isclose(x: Any, y: Any, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Return True if the values x and y are close to each other and False otherwise.

    This is a wrapper around math.isclose() that also works for ag.Scalar objects.
    """
    if isinstance(x, Scalar):
        x_float: float = x.data
    elif isinstance(x, Tensor):
        assert x.is_scalar, f"Logic only implemented for scalars, got shape {x.shape}."
        x_float = x.data[0].data
    else:
        x_float = x
    if isinstance(y, Scalar):
        y_float: float = y.data
    elif isinstance(y, Tensor):
        assert y.is_scalar, f"Logic only implemented for scalars, got shape {y.shape}."
        y_float = y.data[0].data
    else:
        y_float = y
    return math.isclose(x_float, y_float, rel_tol=rel_tol, abs_tol=abs_tol)


def _fill(shape: tuple[int, ...], value: float) -> Tensor:
    """Value `value` in tensor like `x`."""
    data = [value] * math.prod(shape)
    return Tensor(data, shape=shape)


def zeros_like(x: Tensor) -> Tensor:
    """Zeros like `x`."""
    return _fill(x.shape, 0.0)


def ones_like(x: Tensor) -> Tensor:
    """Ones like `x`."""
    return _fill(x.shape, 1.0)


def zeros(shape: tuple[int, ...]) -> Tensor:
    """Zeros of shape `shape`."""
    return _fill(shape, 0.0)


def ones(shape: tuple[int, ...]) -> Tensor:
    """Ones of shape `shape`."""
    return _fill(shape, 1.0)


def arange(start: float, stop: float, step: float = 1) -> Tensor:
    """Return evenly spaced values within a given interval."""
    data = [x for x in range(start, stop, step)]
    return Tensor(data, shape=(len(data),), requires_grad=False, dtype=int)


def maximum(x: Tensor, y: Tensor) -> Tensor:
    """Compute the maximum of two scalars."""
    return x.maximum(y)


def minimum(x: Tensor, y: Tensor) -> Tensor:
    """Compute the minimum of two scalars."""
    return x.minimum(y)


def max(x: Tensor, axis: int = None) -> Tensor:
    """Compute the maximum of a tensor."""
    return x.max(axis)


def min(x: Tensor, axis: int = None) -> Tensor:
    """Compute the minimum of a tensor."""
    return x.min(axis)


def mean(x: Tensor, axis: int = None) -> Tensor:
    """Compute the mean of a tensor."""
    return x.mean(axis)


def clip(x: Tensor, min_value: float, max_value: float) -> Tensor:
    """Clip the value of a scalar between a minimum and a maximum value."""
    return x.clip(min_value, max_value)


def sigmoid(x: Tensor) -> Tensor:
    """Compute the sigmoid of a scalar."""
    return x.sigmoid()


def relu(x: Tensor) -> Tensor:
    """Compute the rectified linear unit of a scalar."""
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    """Compute the hyperbolic tangent of a scalar."""
    return x.tanh()


def exp(x: Tensor) -> Tensor:
    """Compute the exponential of a scalar."""
    return x.exp()


def log(x: Tensor, safe: bool = False) -> Tensor:
    """Compute the natural logarithm of a scalar."""
    if safe:
        return x.maximum(LOG_EPSILON).log()
    return x.log()
