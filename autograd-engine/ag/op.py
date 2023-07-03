"""Operations for the autodiff package."""
import abc
import math
from typing import Any, Tuple, Union

Number = Union[float, int]


# Abstract base class for operations.
class Op(abc.ABC):
    """Abstract base class for operations."""

    NAME: str = "base"

    @abc.abstractmethod
    def forward(self) -> Number:
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self, grad: Number) -> Any:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """


class Add(Op):
    """Add two numbers together."""

    NAME: str = "+"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data + self.y.data

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad, grad


class Mul(Op):
    """Multiply two numbers together."""

    NAME: str = "*"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data * self.y.data

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation."""
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad * self.y.data, grad * self.x.data


class Sub(Op):
    """Subtract two numbers."""

    NAME: str = "-"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data - self.y.data

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad, -grad


class Pow(Op):
    """Raise a number to a power."""

    NAME: str = "**"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data**self.y.data

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return (
            grad * self.y.data * self.x.data ** (self.y.data - 1),
            grad * math.log(self.x.data) * self.x.data**self.y.data,
        )


class Exp(Op):
    """Compute the exponential of a number."""

    NAME: str = "exp"

    def __init__(self, x: "Scalar"):
        """Initialize an operation."""
        self.x = x

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return math.exp(self.x.data)

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad * math.exp(self.x.data)


class Sigmoid(Op):
    """Compute the sigmoid of a number."""

    NAME: str = "sigmoid"

    def __init__(self, x: "Scalar"):
        """Initialize an operation."""
        self.x = x

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return 1 / (1 + math.exp(-self.x.data))

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad * self.forward() * (1 - self.forward())


class Tanh(Op):
    """Compute the hyperbolic tangent of a number."""

    NAME: str = "tanh"

    def __init__(self, x: "Scalar"):
        """Initialize an operation."""
        self.x = x

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return (math.exp(self.x.data) - math.exp(-self.x.data)) / (
            math.exp(self.x.data) + math.exp(-self.x.data)
        )

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad * (1 - self.forward() ** 2)


class Log(Op):
    """Compute the logarithm of a number."""

    NAME: str = "log"

    def __init__(self, x: "Scalar"):
        """Initialize an operation."""
        self.x = x

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return math.log(self.x.data)

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad / self.x.data

    #  def forward(self) -> Number:
    #      """Forward pass of the operation."""
    #      # x = self.x.data
    #      # base = self.y.data
    #      return math.log(self.x.data, self.y.data)
    #
    #  def backward(self) -> Number:
    #      """Backward pass of the operation."""
    #      return 1 / (self.x.data * math.log(self.y.data))


class Max(Op):
    """Compute the maximum of two numbers."""

    NAME: str = "max"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return max(self.x.data, self.y.data)

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad if self.x.data > self.y.data else 0.0


class Min(Op):
    """Compute the minimum of two numbers."""

    NAME: str = "min"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return min(self.x.data, self.y.data)

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad if self.x.data < self.y.data else 0.0


class GreaterThan(Op):
    """Compute whether one number is greater than another."""

    NAME: str = ">"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return 1.0 if self.x.data > self.y.data else 0.0

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return 0.0
