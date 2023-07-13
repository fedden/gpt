"""A nD tensor class for automatic differentiation of scalar functions."""
from __future__ import annotations
import itertools
import math
from typing import List, Optional, Tuple, Union

import numpy as np

import ag
from ag.scalar import Scalar

SCALAR_SHAPE: tuple[int, ...] = tuple()

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
        shape: Optional[tuple[int, ...]] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
        _set_name: bool = True,
        _child_nodes: Optional[list[Tensor]] = None,
    ):
        """Initialize a nD tensor with a value and a gradient."""
        if isinstance(data, Tensor):
            self.data: List[Scalar] = data.data
        elif isinstance(data, np.ndarray):
            self.data = [
                x if isinstance(x, Scalar) else Scalar(x, requires_grad=requires_grad)
                for x in data.flatten()
            ]
        elif isinstance(data, (list, tuple)):
            self.data = [
                x if isinstance(x, Scalar) else Scalar(x, requires_grad=requires_grad)
                for x in flatten(data)
            ]
        else:
            self.data = [
                data
                if isinstance(data, Scalar)
                else Scalar(data, requires_grad=requires_grad)
            ]
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
        self.requires_grad: bool = requires_grad
        self.name: Optional[str] = name

    @property
    def name(self) -> Optional[str]:
        """Return the name of the tensor."""
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        """Set the name of the tensor."""
        self._name = name
        # Set the names of the scalar values.
        if name is not None:
            ranges = [range(dim) for dim in self.shape]
            for nd_i in itertools.product(*ranges):
                flattened_i: int = self._nd_i_to_1d_i(nd_i)
                self.data[flattened_i].name = f"{name}{list(nd_i)}"

    def _nd_i_to_1d_i(self, nd_i: tuple[int, ...]) -> int:
        """Convert multi-dim indices to 1D index for `self.data`."""
        assert len(nd_i) == len(self.shape), (
            f"Number of indices ({len(nd_i)}) does not match number of "
            f"dimensions ({len(self.shape)})."
        )
        index: int = 0
        for dim, dim_i in enumerate(nd_i):
            assert dim_i < self.shape[dim], (
                f"Index {dim_i} is out of bounds for dimension {dim} with "
                f"shape {self.shape[dim]}."
            )
            index *= self.shape[dim]
            index += dim_i
        return index

    @staticmethod
    def _infer_shape(x: AcceptedInput) -> Tuple[int, ...]:
        """Infer the shape of the tensor."""
        if isinstance(x, Tensor):
            return x.shape
        elif isinstance(x, np.ndarray):
            return x.shape
        elif isinstance(x, list):
            # Handle nested lists.
            shape = []
            while isinstance(x, list):
                shape.append(len(x))
                x = x[0]
            return tuple(shape)
        else:
            return SCALAR_SHAPE

    def is_leaf_node(self) -> bool:
        """Return True if the node is a leaf node."""
        return len(self._child_nodes) == 0

    def __repr__(self):
        """Return a string representation of the tensor."""
        object_type: str = self.__class__.__name__
        elements: List[str] = [f"shape={self.shape}"]
        if self.name is not None:
            elements.append(f"name='{self.name}'")
        body: str = ", ".join(elements)
        return f"{object_type}({body})"

    def _int_to_slice(self, index: int, dim: int) -> slice:
        """Convert an integer index to a slice."""
        assert isinstance(index, int), "Index must be an integer."
        assert index < self.shape[dim], "Index out of bounds."
        return slice(index, index + 1, 1)

    def _int_to_tuple(self, index: int) -> tuple:
        """Convert an integer index to a tuple."""
        assert isinstance(index, int), "Index must be an integer."
        assert index < self.shape[0], "Index out of bounds."
        # Index the first dimension, and return slice objects for the rest.
        return (self._int_to_slice(index, dim=0),) + (slice(None, None, None),) * (
            len(self.shape) - 1
        )

    def _slice_to_tuple(self, index: slice) -> tuple:
        """Convert a slice index to a tuple."""
        assert isinstance(index, slice), "Index must be a slice."
        return (index,) + (slice(None, None, None),) * (len(self.shape) - 1)

    def __getitem__(self, key: Union[int, slice, tuple]) -> Tensor:
        """Return a subset of the tensor."""
        # Normalize the index to a tuple of slices.
        if isinstance(key, int):
            per_dim_slice = self._int_to_tuple(key)
        elif isinstance(key, slice):
            per_dim_slice = self._slice_to_tuple(key)
        elif isinstance(key, tuple):
            per_dim_slice = tuple(
                self._int_to_slice(x, dim=dim) if isinstance(x, int) else x
                for dim, x in enumerate(key)
            )
        else:
            raise TypeError("Invalid index type.")
        # Help with type checking.
        assert isinstance(
            per_dim_slice, tuple
        ), f"Invalid index type {type(per_dim_slice)}."
        assert all(
            isinstance(x, slice) for x in per_dim_slice
        ), f"Invalid index type {per_dim_slice}."
        assert len(per_dim_slice) == len(self.shape), "Invalid number of dimensions."
        # Convert the slice to a list of slices.
        int_slices: list[slice] = [
            dim_slice.indices(dim_size)
            for dim_slice, dim_size in zip(per_dim_slice, self.shape)
        ]
        int_ranges: list[range] = [
            range(start, stop, step) for start, stop, step in int_slices
        ]
        # Get the strides for each dimension.
        per_dim_strides: list[int] = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            per_dim_strides[i] = per_dim_strides[i + 1] * self.shape[i + 1]
        sliced_data: list[Scalar] = []
        # Apply the slice to each dimension.
        for nd_i in itertools.product(*int_ranges):
            # Convert the multi-dimensional index to a single index.
            flattened_i: int = self._nd_i_to_1d_i(nd_i)
            sliced_data.append(self.data[flattened_i])
        # Compute the shape of the sliced tensor.
        sliced_shape_list: list[int] = [
            math.ceil((stop - start) / end)  # type: ignore
            for (start, stop, end) in int_slices
        ]
        # Remove any dimensions of size 1.
        sliced_shape: tuple[int, ...] = tuple(x for x in sliced_shape_list if x != 1)
        # Create a new tensor with the sliced data.
        return Tensor(
            data=sliced_data,
            shape=sliced_shape,
            requires_grad=self.requires_grad,
            name=self.name,
        )

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
        )

    def __rsub__(self, other: AcceptedInput) -> Tensor:
        """Subtract two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y - x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
        )

    def __mul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x * y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
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
        )

    def __rtruediv__(self, other: AcceptedInput) -> Tensor:
        """Divide two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y / x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
        )

    def __pow__(self, other: AcceptedInput) -> Tensor:
        """Raise a tensor to a power."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x**y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
        )

    def __rpow__(self, other: AcceptedInput) -> Tensor:
        """Raise a tensor to a power."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[y**x for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
        )

    def __neg__(self) -> Tensor:
        """Negate a tensor."""
        return Tensor(
            data=[-x for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
        )

    @staticmethod
    def _vector_dot(l: Tensor, r: Tensor) -> Scalar:
        """Compute the dot product of two vectors."""
        assert (
            l.shape == r.shape
        ), f"Cannot compute dot product of {l.shape} and {r.shape}."
        assert l.ndim == 1 or l.shape == SCALAR_SHAPE
        assert r.ndim == 1 or r.shape == SCALAR_SHAPE
        return ag.scalar.sum([x * y for x, y in zip(l.data, r.data)])

    @classmethod
    def from_scalar(cls, scalar: Scalar) -> Tensor:
        """Create a tensor from a scalar."""
        return Tensor(
            data=[scalar],
            shape=(1,),
            requires_grad=scalar.requires_grad,
            _child_nodes=[],
        )

    def __matmul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors."""
        l = self
        if isinstance(other, Tensor):
            r = other
        else:
            r = Tensor(data=other)
        if l.ndim == 1 and r.ndim == 1:
            return self._vector_dot(l, r)
        elif l.ndim == 2 and r.ndim == 1:
            assert l.shape[1] == r.shape[0]
            assert l.shape[0] == 1
            dot_product: Scalar = self._vector_dot(l.flatten(), r)
            return self.from_scalar(dot_product)
        elif l.ndim == 1 and r.ndim == 2:
            assert l.shape[0] == r.shape[0]
            assert r.shape[1] == 1
            dot_product: Scalar = self._vector_dot(l, r.flatten())
            return self.from_scalar(dot_product)
        elif l.ndim >= 2 and r.ndim >= 2:
            assert l.shape[-1] == r.shape[-2]
            out_shape = (*l.shape[:-1], r.shape[-1])
            # Collapse all dimensions except the last two into a single
            # dimension, so that we can treat the tensor as a 2D tensor.
            l = l.reshape(-1, l.shape[-2], l.shape[-1])
            r = r.reshape(-1, r.shape[-2], r.shape[-1])
            out_data: list[Scalar] = [None] * math.prod(out_shape)  # type: ignore
            # For each slice of the tensor along the first dimension, take the
            # dot product of each row of the left tensor with each column of
            # the right tensor, and stack the resulting vectors into a new
            # tensor. The shapes of the tensors are:
            # l: (n_slices, n, m)
            # r: (n_slices, m, p)
            # out: (n_slices, n, p)
            n_slices = l.shape[0]
            n_rows = l.shape[1]
            n_cols = r.shape[2]
            for slice_i in range(n_slices):
                for row_i in range(n_rows):
                    for col_i in range(n_cols):
                        l_vector: Tensor = l[slice_i, row_i, :]
                        r_vector: Tensor = r[slice_i, :, col_i]
                        dot_product: Scalar = self._vector_dot(l_vector, r_vector)
                        # The following line is equivalent to:
                        # out_data[slice_i, row_i, col_i] = dot_product
                        out_data[
                            slice_i * n_rows * n_cols + row_i * n_cols + col_i
                        ] = dot_product
            return Tensor(
                data=out_data,
                shape=out_shape,
                requires_grad=l.requires_grad or r.requires_grad,
                _child_nodes=[l, r],
            )
        else:
            raise ValueError(
                f"Cannot perform matrix multiplication on tensors with "
                f"shapes {l.shape} and {r.shape}."
            )

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array."""
        return np.array([x.data for x in self.data]).reshape(self.shape)

    @property
    def size(self) -> int:
        """Return the number of elements in the tensor."""
        return len(self.data)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions in the tensor."""
        return len(self.shape)

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

    def flatten(self) -> Tensor:
        """Flatten the tensor."""
        return self.reshape(-1)

    def repeat(self, repeats: int) -> Tensor:
        """Repeat each element of the tensor after themselves."""
        assert isinstance(repeats, int), "repeats must be an int."
        assert repeats > 0, "repeats must be positive."
        new_shape = (self.size * repeats,)
        new_data = [None] * new_shape[0]
        for i in range(self.size):
            for j in range(repeats):
                new_data[i * repeats + j] = self.data[i]  # type: ignore
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
        )

    def tanh(self) -> Tensor:
        """Apply the tanh function to the tensor."""
        return Tensor(
            data=[x.tanh() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
        )

    def relu(self) -> Tensor:
        """Apply the relu function to the tensor."""
        return Tensor(
            data=[x.relu() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
        )

    def mean(self, axis: Optional[int] = None) -> Tensor:
        """Compute the mean of the tensor."""
        if axis is None:
            # Compute the mean over all axes.
            return Tensor(
                data=ag.scalar.sum(self.data) / self.size,
                shape=SCALAR_SHAPE,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        elif isinstance(axis, int):
            raise NotImplementedError("Regression check against NumPy/Torch")
            assert 0 <= axis < len(self.shape), "axis out of bounds."
            new_shape = tuple(x for i, x in enumerate(self.shape) if i != axis)
            new_data = []
            for i in range(self.size // self.shape[axis]):
                new_data.append(
                    ag.scalar.sum(
                        self.data[i * self.shape[axis] : (i + 1) * self.shape[axis]]
                    )
                    / self.shape[axis]
                )
            return Tensor(
                data=new_data,
                shape=new_shape,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        elif isinstance(axis, (list, tuple)):
            assert all(isinstance(x, int) for x in axis), "axis must be a list of ints."
            assert all(0 <= x < len(self.shape) for x in axis), "axis out of bounds."
            new_shape = tuple(x for i, x in enumerate(self.shape) if i not in axis)
            new_data = []
            for i in range(self.size // math.prod([self.shape[x] for x in axis])):
                new_data.append(
                    ag.scalar.sum(
                        self.data[
                            i
                            * math.prod([self.shape[x] for x in axis]) : (i + 1)
                            * math.prod([self.shape[x] for x in axis])
                        ]
                    )
                    / math.prod([self.shape[x] for x in axis])
                )
            return Tensor(
                data=new_data,
                shape=new_shape,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        else:
            raise TypeError("axis must be an int or a list of ints.")

    @property
    def grad(self) -> Tensor:
        """Return the gradient of the tensor."""
        assert self.requires_grad, "This tensor does not require gradients."
        return Tensor(
            data=[x.grad for x in self.data],
            shape=self.shape,
            requires_grad=False,
            _child_nodes=[],
        )

    def backward(self):
        """Backpropagate the gradient through the graph."""
        # First ensure we have a scalar.
        assert (
            self.size == 1
        ), f"Cannot call backward on a non-scalar tensor of size {self.size}."
        self.data[0].backward()

    def zero_grad(self):
        """Zero any gradients."""
        for scalar in self.data:
            scalar.grad.data = 0.0


def _to_elementwise_op_tensor(x: Tensor, y: AcceptedInput) -> tuple[Tensor, Tensor]:
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
