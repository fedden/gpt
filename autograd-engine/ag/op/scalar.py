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


class Identity(Op):
    """Identity operation."""

    NAME: str = "identity"

    def __init__(self, x: "Scalar"):
        """Initialize an operation."""
        self.x = x

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data

    def backward(self, grad: Number) -> Number:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return grad


class Pow(Op):
    """Raise a number to a power."""

    NAME: str = "**"

    def __init__(self, base: "Scalar", exponent: "Scalar"):
        """Initialize an operation."""
        self.base = base
        self.exponent = exponent

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.base.data**self.exponent.data

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        grad_base: Number = (
            grad * self.exponent.data * self.base.data ** (self.exponent.data - 1)
        )
        if self.base.data > 0:
            grad_exponent: Number = (
                grad * self.base.data**self.exponent.data * math.log(self.base.data)
            )
        else:
            # If the base is negative or zero, the gradient for the exponent
            # would be a complex number. We set it to NaN to indicate that it
            # is not defined, rather than raising an exception, and hopefully
            # the user will notice that something is wrong and fix their usage
            # of the library.
            grad_exponent = float("nan")
        return grad_base, grad_exponent


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


class Maximum(Op):
    """Compute the maximum of two numbers."""

    NAME: str = "maximum"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return max(self.x.data, self.y.data)

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        if self.x.data > self.y.data:
            return grad, 0.0
        else:
            return 0.0, grad


class Minimum(Op):
    """Compute the minimum of two numbers."""

    NAME: str = "minimum"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return min(self.x.data, self.y.data)

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        if self.x.data < self.y.data:
            return grad, 0.0
        else:
            return 0.0, grad


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

    def backward(self, grad: Number) -> Tuple[Number, Number]:
        """Backward pass of the operation.

        Parameters
        ----------
        grad: float
            The gradient of the loss with respect to the output of the
            operation.
        """
        return 0.0, 0.0
