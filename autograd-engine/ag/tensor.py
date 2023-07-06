"""A nD tensor class for automatic differentiation of scalar functions."""
from __future__ import annotations
import itertools
import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np

import ag
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
            self.data: List[Scalar] = data.data
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
        for multi_dim_i in itertools.product(*int_ranges):
            # Convert the multi-dimensional index to a single index.
            single_dim_i: int = sum(
                multi_dim_i[i] * per_dim_strides[i] for i in range(self.ndim)
            )
            sliced_data.append(self.data[single_dim_i])
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

    @staticmethod
    def _vector_dot(l: Tensor, r: Tensor) -> Scalar:
        """Compute the dot product of two vectors."""
        assert l.shape == r.shape, f"Cannot compute dot product of {l.shape} and {r.shape}."
        assert l.ndim == 1
        assert r.ndim == 1
        return ag.sum([x * y for x, y in zip(l.data, r.data)])

    @classmethod
    def from_scalar(cls, scalar: Scalar) -> Tensor:
        """Create a tensor from a scalar."""
        return Tensor(
            data=[scalar],
            shape=(1,),
            requires_grad=scalar.requires_grad,
            _child_nodes=[],
            _op_type="scalar",
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
            # For 2D tensors, e.g with shape (2, 3) and (3, 4), we need to do
            # the following:
            # 1. Take the dot product of each row of the left tensor with each
            #    column of the right tensor.
            # 2. Stack the resulting vectors into a new tensor.
            # For example, if we have:
            # l = [[1, 2, 3], [4, 5, 6]]
            # r = [[7, 8], [9, 10], [11, 12]]
            # Then we need to do:
            # l[0] dot r[:, 0] = [1, 2, 3] dot [7, 9, 11] = 1 * 7 + 2 * 9 + 3 * 11
            # l[0] dot r[:, 1] = [1, 2, 3] dot [8, 10, 12] = 1 * 8 + 2 * 10 + 3 * 12
            # l[1] dot r[:, 0] = [4, 5, 6] dot [7, 9, 11] = 4 * 7 + 5 * 9 + 6 * 11
            # l[1] dot r[:, 1] = [4, 5, 6] dot [8, 10, 12] = 4 * 8 + 5 * 10 + 6 * 12
            # And then stack the resulting vectors into a new tensor:
            # [[1 * 7 + 2 * 9 + 3 * 11, 1 * 8 + 2 * 10 + 3 * 12],
            #  [4 * 7 + 5 * 9 + 6 * 11, 4 * 8 + 5 * 10 + 6 * 12]]
            #
            # For tensors with more than 2 dimensions, we need to do the same
            # thing, but for each "slice" of the tensor along each of the
            # proceeding dimensions.

            # Old code that only did the above for 2D tensors:
            #  assert l.shape[-1] == r.shape[-2]
            #  out_shape = (*l.shape[:-1], r.shape[-1])
            #  out_data: list[Scalar] = [None] * (out_shape[0] * out_shape[1])  # type: ignore
            #  for i in range(l.shape[1]):
            #      vector_product: list[Scalar] = self._vector_dot(l[:, i], r[i, :]).data
            #      for j in range(len(vector_product)):
            #          out_data[j * out_shape[1] + i] = vector_product[j]
            #  return Tensor(
            #      data=out_data,
            #      shape=out_shape,
            #      requires_grad=l.requires_grad or r.requires_grad,
            #      _child_nodes=[l, r],
            #      _op_type="matmul",
            #  )

            # New code that does the above for tensors with any number of
            # dimensions:
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
                        out_data[slice_i * n_rows * n_cols + row_i * n_cols + col_i] = dot_product
            return Tensor(
                data=out_data,
                shape=out_shape,
                requires_grad=l.requires_grad or r.requires_grad,
                _child_nodes=[l, r],
                _op_type="matmul",
            )
        else:
            raise ValueError(
                f"Cannot perform matrix multiplication on tensors with shapes {l.shape} and {r.shape}."
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
