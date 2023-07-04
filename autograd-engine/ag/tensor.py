"""A nD tensor class for automatic differentiation of scalar functions."""
from __future__ import annotations
import copy
import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from ag import Scalar

AcceptedInput = Union[Scalar, float, int, list, np.ndarray]


def flatten(container):
    """Flatten a nested container."""
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i


class Tensor:
    """A nD tensor class for automatic differentiation of scalar functions."""

    def __init__(
        self,
        data: AcceptedInput,
        shape: Optional[Tuple[int, ...]] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
        _child_nodes: Optional[List[Tensor]] = None,
        _op_type: Optional[Type["Uknown"]] = None,
    ):
        """Initialize a nD tensor with a value and a gradient."""
        if isinstance(data, Tensor):
            self.data: List[Scalar] = copy.deepcopy(data.data)
        elif isinstance(data, np.ndarray):
            self.data = [Scalar(x, requires_grad=requires_grad) for x in data.flatten()]
        elif isinstance(data, list):
            self.data = [Scalar(x, requires_grad=requires_grad) for x in flatten(data)]
        else:
            self.data = [Scalar(data, requires_grad=requires_grad)]
        self.requires_grad: bool = requires_grad
        self.grad: List[float] = [0.0] * len(self.data)
        self.name: Optional[str] = name
        if shape is None:
            self.shape: Tuple[int, ...] = self._infer_shape(data)
        else:
            self.shape = shape
        assert len(self.data) == math.prod(
            self.shape
        ), f"Shape {self.shape} does not match data length {len(self.data)}."
        if _child_nodes is None:
            self._child_nodes: List[Tensor] = []
        else:
            assert isinstance(_child_nodes, list), "_child_nodes must be a list."
            assert all(
                isinstance(node, Tensor) for node in _child_nodes
            ), "_child_nodes must be a list of Tensors."
            self._child_nodes = _child_nodes
        self._op_type: Optional[Type[Op]] = _op_type

    @staticmethod
    def _infer_shape(x: AcceptedInput) -> Tuple[int, ...]:
        """Infer the shape of the tensor."""
        if isinstance(x, Tensor):
            return x.shape
        if isinstance(x, np.ndarray):
            return x.shape
        if isinstance(x, list):
            # Handle nested lists.
            shape = []
            while isinstance(x, list):
                shape.append(len(x))
                x = x[0]
            return tuple(shape)
        return (1,)

    def is_leaf_node(self) -> bool:
        """Return True if the node is a leaf node."""
        return len(self._child_nodes) == 0 and self._op_type is None

    def __repr__(self):
        """Return a string representation of the tensor."""
        object_type: str = self.__class__.__name__
        elements: List[str] = [f"shape={self.shape}"]
        if self.name is not None:
            elements.append(f"name='{self.name}'")
        body: str = ", ".join(elements)
        return f"{object_type}({body})"

    def __getitem__(self, key: Union[int, slice, tuple]) -> Tensor:
        """Return a subset of the tensor."""
        if isinstance(key, int):
            assert key < len(self.data), "Index out of bounds."
            return Tensor(
                data=self.data[key],
                shape=(1,),
                requires_grad=self.requires_grad,
                name=self.name,
            )
        elif isinstance(key, slice):
            return Tensor(
                data=self.data[key],
                shape=self.shape[key],
                requires_grad=self.requires_grad,
                name=self.name,
            )
        elif isinstance(key, tuple):
            assert len(key) == len(self.shape), "Invalid number of dimensions."
            assert all(isinstance(x, (int, slice)) for x in key), "Invalid index type."
            return Tensor(
                data=self.data[key],
                shape=self.shape[key],
                requires_grad=self.requires_grad,
                name=self.name,
            )
        else:
            raise TypeError("Invalid index type.")

    def broadcast_to_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Broadcast the tensor to a new shape."""
        assert isinstance(
            shape, (list, tuple)
        ), f"shape must be a sequence, not {type(shape)}."
        assert all(isinstance(x, int) for x in shape), "shape must be a tuple of ints."
        assert np.prod(shape) == np.prod(
            self.shape
        ), f"Cannot broadcast shape {self.shape} to shape {shape}."
        return Tensor(
            data=self.data,
            shape=shape,
            requires_grad=self.requires_grad,
            name=self.name,
        )

    def __add__(self, other: AcceptedInput) -> Tensor:
        """Add two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x + y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="add",
        )

    def __radd__(self, other: AcceptedInput) -> Tensor:
        """Add two tensors."""
        return self + other

    def __sub__(self, other: AcceptedInput) -> Tensor:
        """Subtract two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x - y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="sub",
        )

    def __rsub__(self, other: AcceptedInput) -> Tensor:
        """Subtract two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y - x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="sub",
        )

    def __mul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x * y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="mul",
        )

    def __rmul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors."""
        return self * other

    def __truediv__(self, other: AcceptedInput) -> Tensor:
        """Divide two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x / y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="div",
        )

    def __rtruediv__(self, other: AcceptedInput) -> Tensor:
        """Divide two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y / x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="div",
        )

    def __pow__(self, other: AcceptedInput) -> Tensor:
        """Raise a tensor to a power."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x**y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="pow",
        )

    def __rpow__(self, other: AcceptedInput) -> Tensor:
        """Raise a tensor to a power."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y**x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
            _op_type="pow",
        )

    def __neg__(self) -> Tensor:
        """Negate a tensor."""
        return Tensor(
            data=[-x for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
            _op_type="neg",
        )

    def __matmul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors."""
        breakpoint()

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array."""
        return np.array([x.data for x in self.data]).reshape(self.shape)

    @property
    def size(self) -> int:
        """Return the number of elements in the tensor."""
        return len(self.data)

    def tile(self, reps: List[int]) -> Tensor:
        """Tile the tensor."""
        assert isinstance(reps, list), "reps must be a list."
        assert all(isinstance(x, int) for x in reps), "reps must be a list of ints."
        assert len(reps) == len(self.shape), "The number of dimensions must match."
        result_final_shape = tuple(s * t for s, t in zip(self.shape, reps))
        result = self
        n: int = self.size
        for dimension_size, n_reps in zip(self.shape, reps):
            if n_reps != 1:
                result = result.reshape(-1, n)
                result = result.repeat(n_reps)
            n //= dimension_size
        return result.reshape(result_final_shape)

    def reshape(self, *args) -> Tensor:
        """Reshape the tensor."""
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        # Handle -1.
        if -1 in shape:
            assert shape.count(-1) == 1, "Only one -1 is allowed."
            expanded_shape = list(shape)
            expanded_shape[shape.index(-1)] = math.prod(self.shape) // math.prod(
                [x for x in shape if x != -1]
            )
            shape = tuple(expanded_shape)
        assert isinstance(shape, (list, tuple)), "shape must be a tuple."
        assert all(
            isinstance(x, int) for x in shape
        ), "shape must be a sequence of ints."
        assert math.prod(shape) == math.prod(
            self.shape
        ), f"Cannot reshape shape {self.shape} to shape {shape}."
        return Tensor(
            data=self.data,
            shape=tuple(shape),
            requires_grad=self.requires_grad,
            name=self.name,
        )

    def repeat(self, repeats: int) -> Tensor:
        """Repeat each element of the tensor after themselves."""
        assert isinstance(repeats, int), "repeats must be an int."
        assert repeats > 0, "repeats must be positive."
        new_shape = (self.size * repeats,)
        new_data = [None] * new_shape[0]
        for i in range(self.size):
            for j in range(repeats):
                new_data[i * repeats + j] = copy.deepcopy(self.data[i])  # type: ignore
        return Tensor(
            data=new_data,
            shape=new_shape,
            requires_grad=self.requires_grad,
            name=self.name,
        )

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function to the tensor."""
        return Tensor(
            data=[x.sigmoid() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
            _op_type="sigmoid",
        )

    def tanh(self) -> Tensor:
        """Apply the tanh function to the tensor."""
        return Tensor(
            data=[x.tanh() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
            _op_type="tanh",
        )

    def relu(self) -> Tensor:
        """Apply the relu function to the tensor."""
        return Tensor(
            data=[x.relu() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
            _op_type="relu",
        )


def _to_elementwise_op_tensor(x: Tensor, y: AcceptedInput):
    """Ensure y is a Tensor, and broadcast x and y to the same shape."""
    if not isinstance(y, Tensor):
        y = Tensor(y)
    assert isinstance(y, Tensor), "y must be a Tensor."
    # Handle scalar.
    if x.size == 1:
        x = x.reshape([1 for _ in y.shape])
    if y.size == 1:
        y = y.reshape([1 for _ in x.shape])
    assert len(x.shape) == len(y.shape), "The number of dimensions must match."
    if x.shape != y.shape:
        x_tile_reps = [1] * len(x.shape)
        y_tile_reps = [1] * len(y.shape)
        for dim_i, (x_dim, y_dim) in enumerate(zip(x.shape, y.shape)):
            if x_dim == y_dim:
                continue
            elif x_dim == 1:
                x_tile_reps[dim_i] = y_dim
            elif y_dim == 1:
                y_tile_reps[dim_i] = x_dim
            else:
                raise ValueError(f"Cannot broadcast shapes {x.shape} and {y.shape}.")
        if math.prod(x_tile_reps) > 1:
            x = x.tile(x_tile_reps)
        if math.prod(y_tile_reps) > 1:
            y = y.tile(y_tile_reps)
    return x, y
