"""A scalar value with a gradient."""
from __future__ import annotations
from typing import Any, List, Optional, Type, Tuple, Union

import numpy as np

from ag.op.scalar import Add, Mul, Op, Pow, Exp, Sigmoid, Tanh, Log, Sub, Max, Min, GreaterThan


Number = Union[float, int]
AcceptedInput = Union[Number, "Scalar"]


class Scalar:
    """A scalar value with a gradient."""

    def __init__(
        self,
        data: AcceptedInput,
        requires_grad: bool = False,
        name: Optional[str] = None,
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
        self.name: Optional[str] = name
        if _child_nodes is None:
            self._child_nodes: List[Scalar] = []
        else:
            assert isinstance(_child_nodes, list), "_child_nodes must be a list."
            assert all(
                isinstance(node, Scalar) for node in _child_nodes
            ), "_child_nodes must be a list of Scalars."
            self._child_nodes = _child_nodes
        self._op_type: Optional[Type[Op]] = _op_type

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
        elements: List[str] = [f"{self.data:.4f}", f"grad={self.grad:.4f}"]
        if self.name is not None:
            elements.append(f"name='{self.name}'")
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
        """Divide two scalars.

        self / other
        """
        return self * other**-1

    def __rtruediv__(self, other: AcceptedInput) -> Scalar:
        """Divide two scalars.

        other / self
        """
        return Scalar(other) * self**-1

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

    def min(self, other: AcceptedInput) -> Scalar:
        """Compute the minimum of two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Min(self, other).forward(), _child_nodes=[self, other], _op_type=Min
        )

    def clip(self, min_value: AcceptedInput, max_value: AcceptedInput) -> Scalar:
        """Clip a scalar."""
        return self.max(min_value).min(max_value)

    def exp(self) -> Scalar:
        """Compute the exponential of a scalar."""
        return Scalar(data=Exp(self).forward(), _child_nodes=[self], _op_type=Exp)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of a scalar."""
        return Scalar(
            data=Sigmoid(self).forward(), _child_nodes=[self], _op_type=Sigmoid
        )

    def tanh(self) -> Scalar:
        """Compute the hyperbolic tangent of a scalar."""
        return Scalar(data=Tanh(self).forward(), _child_nodes=[self], _op_type=Tanh)

    def relu(self) -> Scalar:
        """Compute the rectified linear unit of a scalar."""
        return self.max(0)

    def log(self) -> Scalar:
        """Compute the natural logarithm of a scalar."""
        return Scalar(data=Log(self).forward(), _child_nodes=[self], _op_type=Log)

    def backward(self, grad: Optional[float] = None) -> None:
        """Backward pass through the graph."""
        # If this is the first node, set the gradient to 1.0.
        if grad is None:
            # Set the gradient for the first node to 1.0 because the gradient
            # of a scalar with respect to itself is 1.0.
            grad = self.grad = 1.0
        else:
            # Only add to the gradient if the node requires a gradient.
            self.grad += grad
        # If this is a leaf node, we're done.
        if self.is_leaf_node():
            # This is a leaf node.
            return
        # Otherwise, this is an internal node, and an operation must have been
        # performed on the child nodes.
        assert self._op_type is not None, "Operation type must be defined."
        # Create the operation, passing in the child nodes. We will use this
        # operation to compute the gradient for each child node.
        op: Op = self._op_type(*self._child_nodes)  # type: ignore
        # First, compute the gradient for each child node, this function may
        # return one or a tuple of gradients, depending on the operation, so we
        # need to handle both cases by converting the result to a tuple.
        child_grads: Tuple[float, ...] = _wrap_as_tuple(op.backward(grad))
        # Ensure the op is implemented correctly, or atleast that it returns
        # the correct number of gradients.
        assert len(child_grads) == len(self._child_nodes), (
            f"Expected {len(self._child_nodes)} gradients, "
            f"but got {len(child_grads)}, for operation {op}."
        )
        # Propagate the gradient to each child node.
        for child_node, child_grad in zip(self._child_nodes, child_grads):
            child_node.backward(child_grad)


class Parameter(Scalar):
    """A scalar value with a gradient that can be optimized."""

    def __init__(self, data: AcceptedInput, *args, **kwargs):
        """Initialize a scalar value with a gradient that can be optimized."""
        super().__init__(data, *args, **kwargs, requires_grad=True)  # type: ignore


def _wrap_as_tuple(x: Any) -> Tuple[Any, ...]:
    """Wrap a value as a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)
