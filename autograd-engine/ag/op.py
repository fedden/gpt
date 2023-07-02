"""Operations for the autodiff package."""
import abc
import math
from typing import Union

Number = Union[float, int]


# Abstract base class for operations.
class Op(abc.ABC):
    """Abstract base class for operations."""

    NAME: str = "base"

    def __init__(self, x: "Scalar", y: "Scalar"):
        """Initialize an operation."""
        self.x = x
        self.y = y

    @abc.abstractmethod
    def forward(self) -> Number:
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self) -> Number:
        """Backward pass of the operation."""


class Add(Op):
    """Add two numbers together."""

    NAME: str = "+"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data + self.y.data

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return 1.0


class Mul(Op):
    """Multiply two numbers together."""

    NAME: str = "*"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data * self.y.data

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return self.y.data


class Div(Op):
    """Divide two numbers."""

    NAME: str = "/"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data / self.y.data

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return 1 / self.y.data


class Sub(Op):
    """Subtract two numbers."""

    NAME: str = "-"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data - self.y.data

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return -1.0


class Pow(Op):
    """Raise a number to a power."""

    NAME: str = "**"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return self.x.data**self.y.data

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return self.y.data * self.x.data ** (self.y.data - 1)


class Log(Op):
    """Compute the logarithm of a number."""

    def forward(self) -> Number:
        """Forward pass of the operation."""
        # x = self.x.data
        # base = self.y.data
        return math.log(self.x.data, self.y.data)

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return 1 / (self.x.data * math.log(self.y.data))


class Max(Op):
    """Compute the maximum of two numbers."""

    NAME: str = "max"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return max(self.x.data, self.y.data)

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return 1.0 if self.x.data > self.y.data else 0.0


class GreaterThan(Op):
    """Compute whether one number is greater than another."""

    NAME: str = ">"

    def forward(self) -> Number:
        """Forward pass of the operation."""
        return 1.0 if self.x.data > self.y.data else 0.0

    def backward(self) -> Number:
        """Backward pass of the operation."""
        return 0.0
