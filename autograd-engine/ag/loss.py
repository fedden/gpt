"""Loss functions."""
import ag
from ag.scalar import Scalar


def hinge(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the hinge loss."""
    return ag.max(Scalar(0), Scalar(1) - y_true * y_pred)


def mse(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the mean squared error."""
    return ag.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the binary cross-entropy."""
    return -y_true * ag.log(y_pred, safe=True) - (Scalar(1) - y_true) * ag.log(
        Scalar(1) - y_pred, safe=True
    )
