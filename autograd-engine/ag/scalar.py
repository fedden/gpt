"""A scalar value with a gradient."""
from __future__ import annotations
import math
from typing import List, Optional, Type, Union

import numpy as np

from ag.op import Add, Div, Mul, Op, Pow, Log, Sub, Max, GreaterThan


Number = Union[float, int]
AcceptedInput = Union[Number, "Scalar"]


class Scalar:
    """A scalar value with a gradient."""

    def __init__(
        self,
        data: AcceptedInput,
        requires_grad: bool = False,
        _child_nodes: Optional[List[Scalar]] = None,
        _op_type: Optional[Type[Op]] = None,
    ):
        """Initialize a scalar value with a gradient."""
        if isinstance(data, Scalar):
            self.data: float = data.data
        else:
            # Check data type is numeric.
            assert isinstance(
                data, (float, int, np.integer, np.floating)
            ), f"Invalid data type: {type(data)}"
            self.data = float(data)
        self.requires_grad: bool = requires_grad
        self.grad: float = 0.0
        if _child_nodes is None:
            self._child_nodes = []
        else:
            assert isinstance(_child_nodes, list), "_child_nodes must be a list."
            assert all(
                isinstance(node, Scalar) for node in _child_nodes
            ), "_child_nodes must be a list of Scalars."
            self._child_nodes = _child_nodes
        self._op_type = _op_type

    @staticmethod
    def _to_scalar(x: AcceptedInput) -> Scalar:
        """Convert a non-scalar to a scalar."""
        if isinstance(x, Scalar):
            return x
        return Scalar(x)

    def is_leaf_node(self) -> bool:
        """Return True if the node is a leaf node."""
        return len(self._child_nodes) == 0 and self._op_type is None

    def __repr__(self):
        """Return a string representation of the scalar."""
        object_type: str = self.__class__.__name__
        elements: List[str] = [f"{self.data:.4f}"]
        if self.requires_grad:
            elements.append(f"grad={self.grad:.4f}")
        if not self.is_leaf_node():
            assert self._op_type is not None, "Operation type must be defined."
            assert len(self._child_nodes) == 2, "Invalid number of child nodes."
            x_node, y_node = self._child_nodes
            elements.append(f"from=({x_node} {self._op_type.NAME} {y_node}))")
        body: str = ", ".join(elements)
        return f"{object_type}({body})"

    def __add__(self, other: AcceptedInput) -> Scalar:
        """Add two scalars together."""
        other = self._to_scalar(other)
        return Scalar(
            data=Add(self, other).forward(), _child_nodes=[self, other], _op_type=Add
        )

    def __mul__(self, other: AcceptedInput) -> Scalar:
        """Multiply two scalars together."""
        other = self._to_scalar(other)
        return Scalar(
            data=Mul(self, other).forward(), _child_nodes=[self, other], _op_type=Mul
        )

    def __truediv__(self, other: AcceptedInput) -> Scalar:
        """Divide two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Div(self, other).forward(), _child_nodes=[self, other], _op_type=Div
        )

    def __sub__(self, other: AcceptedInput) -> Scalar:
        """Subtract two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Sub(self, other).forward(), _child_nodes=[self, other], _op_type=Sub
        )

    def __pow__(self, other: AcceptedInput) -> Scalar:
        """Raise a scalar to a power."""
        other = self._to_scalar(other)
        return Scalar(
            data=Pow(self, other).forward(), _child_nodes=[self, other], _op_type=Pow
        )

    def __neg__(self) -> Scalar:
        """Negate a scalar."""
        return Scalar(0.0) - self

    def __gt__(self, other: AcceptedInput) -> Scalar:
        """Compare two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=GreaterThan(self, other).forward(),
            _child_nodes=[self, other],
            _op_type=GreaterThan,
        )

    def max(self, other: AcceptedInput) -> Scalar:
        """Compute the maximum of two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Max(self, other).forward(), _child_nodes=[self, other], _op_type=Max
        )

    def exp(self) -> Scalar:
        """Compute the exponential of a scalar."""
        e: Scalar = Scalar(math.e)
        return Scalar(data=Pow(e, self).forward(), _child_nodes=[e, self], _op_type=Pow)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of a scalar."""
        one: Scalar = Scalar(1.0)
        return one / (one + (-self).exp())

    def tanh(self) -> Scalar:
        """Compute the hyperbolic tangent of a scalar."""
        x = (self * 2).exp()
        return (x - 1) / (x + 1)

    def relu(self) -> Scalar:
        """Compute the rectified linear unit of a scalar."""
        return self.max(0)

    def log(self) -> Scalar:
        """Compute the natural logarithm of a scalar."""
        # The base of the natural logarithm.
        e: Scalar = Scalar(math.e)
        return Scalar(data=Log(self, e).forward(), _child_nodes=[self, e], _op_type=Log)

    def backward(self, grad: float = 1.0) -> None:
        """Backward pass through the graph."""
        # Set the gradient for this node, defaulting to 1.0 because the
        # gradient of a scalar with respect to itself is 1.0.
        if self.requires_grad:
            self.grad += grad
        # If this is a leaf node, we're done.
        if self.is_leaf_node():
            # This is a leaf node.
            return
        # Otherwise, this is an internal node.
        assert self._op_type is not None, "Operation type must be defined."
        assert len(self._child_nodes) == 2, "Invalid number of child nodes."
        x_node, y_node = self._child_nodes
        # Compute the local gradient for each child node.
        x_grad: Number = self._op_type(x_node, y_node).backward()
        y_grad: Number = self._op_type(y_node, x_node).backward()
        # Propagate the gradient to each child node.
        x_node.backward(x_grad * grad)
        y_node.backward(y_grad * grad)


class Parameter(Scalar):
    """A scalar value with a gradient that can be optimized."""

    def __init__(self, data: AcceptedInput):
        """Initialize a scalar value with a gradient that can be optimized."""
        super().__init__(data, requires_grad=True)
