"""Loss functions."""
import ag
from ag.scalar import Scalar


def hinge(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the hinge loss."""
    return ag.max(Scalar(0), Scalar(1) - y_true * y_pred)


def mse(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the mean squared error."""
    return (y_true - y_pred) ** 2


def binary_cross_entropy(y_true: Scalar, y_pred: Scalar) -> Scalar:
    """Compute the binary cross-entropy."""
    return -y_true * ag.log(y_pred) - (Scalar(1) - y_true) * ag.log(Scalar(1) - y_pred)
    #  epsilon = 1e-9
    #  y_pred = ag.clip(y_pred, epsilon, 1.0 - epsilon)
    #  log_probability = ag.log(y_pred)
    #  log_complement_probability = ag.log(Scalar(1.0) - y_pred)
    #  term_1 = y_true * log_probability
    #  term_2 = (Scalar(1.0) - y_true) * log_complement_probability
    #  loss = term_1 + term_2
    #  return -loss
