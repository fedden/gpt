"""A scalar value with a gradient."""
from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ag.op.scalar import (
    Add,
    Exp,
    GreaterThan,
    Identity,
    Log,
    Maximum,
    Minimum,
    Mul,
    Op,
    Pow,
    Tanh,
    Sigmoid,
    Sub,
)


Number = Union[float, int]
AcceptedInput = Union[Number, "Scalar"]


class Scalar:
    """A scalar value with a gradient."""

    _no_grad: bool = False

    def __init__(
        self,
        data: AcceptedInput,
        requires_grad: bool = False,
        name: Optional[str] = None,
        dtype: Optional[type] = float,
        _set_gradient: bool = True,
        _child_nodes: Optional[list[Scalar]] = None,
        _op_type: Optional[type[Op]] = None,
    ):
        """Initialize a scalar value with a gradient."""
        if isinstance(data, Scalar):
            raise ValueError(
                "Copy constructors break the graph, use `Scalar.identity` instead."
            )
        else:
            # Check data type is numeric.
            assert isinstance(
                data, (float, int, np.integer, np.floating)
            ), f"Invalid data type: {type(data)}"
            # Check dtype is valid.
            assert dtype in (float, int, bool), f"Invalid dtype: {dtype}"
            self.dtype = dtype
            self.data = dtype(data)
            self.requires_grad = requires_grad
            # TODO(leonfedden): Handle 2nd/3rd order gradients - recursion?
            if _set_gradient:
                self.grad = Scalar(0.0, requires_grad=False, _set_gradient=False)
            else:
                self.grad = None
            self.name = name
            if _child_nodes is None or self._no_grad:
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
        return len(self._child_nodes) == 0

    def __repr__(self):
        """Return a string representation of the scalar."""
        object_type: str = self.__class__.__name__
        elements: List[str] = [f"{self.data:.4f}"]
        if self.grad is not None:
            elements.append(f"grad={self.grad.data:.4f}")
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

    def maximum(self, other: AcceptedInput) -> Scalar:
        """Compute the maximum of two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Maximum(self, other).forward(),
            _child_nodes=[self, other],
            _op_type=Maximum,
        )

    def minimum(self, other: AcceptedInput) -> Scalar:
        """Compute the minimum of two scalars."""
        other = self._to_scalar(other)
        return Scalar(
            data=Minimum(self, other).forward(),
            _child_nodes=[self, other],
            _op_type=Minimum,
        )

    def clip(self, min_value: AcceptedInput, max_value: AcceptedInput) -> Scalar:
        """Clip a scalar."""
        return self.maximum(min_value).minimum(max_value)

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
        return self.maximum(0)

    def log(self) -> Scalar:
        """Compute the natural logarithm of a scalar."""
        return Scalar(data=Log(self).forward(), _child_nodes=[self], _op_type=Log)

    def identity(self) -> Scalar:
        """Return the scalar."""
        return Scalar(
            data=Identity(self).forward(), _child_nodes=[self], _op_type=Identity
        )

    def numpy(self) -> np.ndarray:
        """Return the scalar as a numpy array."""
        return np.array(self.data)

    def backward(self, grad: Optional[float] = None) -> None:
        """Backward pass through the graph."""
        assert self.grad is not None, "Gradient must be defined."
        # If this is the first node, set the gradient to 1.0.
        if grad is None:
            # Set the gradient for the first node to 1.0 because the gradient
            # of a scalar with respect to itself is 1.0.
            # TODO(leonfedden): Handle 2nd/3rd order gradients.
            grad = self.grad = Scalar(1.0, requires_grad=False, _set_gradient=False)
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

    def int(self) -> Scalar:
        """Cast the scalar to an integer."""
        return Scalar(data=int(self.data), dtype=int)

    def float(self) -> Scalar:
        """Cast the scalar to a float."""
        return Scalar(data=float(self.data), dtype=float)

    def bool(self) -> Scalar:
        """Cast the scalar to a bool."""
        return Scalar(data=bool(self.data), dtype=bool)


def _wrap_as_tuple(x: Any) -> Tuple[Any, ...]:
    """Wrap a value as a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)
