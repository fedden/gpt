"""Loss functions."""
import ag
from ag.tensor import Tensor


def hinge(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the hinge loss."""
    return ag.maximum(Tensor(0), Tensor(1) - y_true * y_pred)


def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the mean squared error."""
    return ag.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the binary cross-entropy."""
    return -y_true * ag.log(y_pred, safe=True) - (Tensor(1) - y_true) * ag.log(
        Tensor(1) - y_pred, safe=True
    )
