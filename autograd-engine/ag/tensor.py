"""A nD tensor class for automatic differentiation of scalar functions."""
from __future__ import annotations
import itertools
import math
from typing import Optional, Union

import numpy as np

import ag
from ag.scalar import Scalar

SCALAR_SHAPE: tuple[int, ...] = tuple()

AcceptedInput = Union[Scalar, float, int, list, np.ndarray]


class Tensor:
    """A nD tensor class for automatic differentiation of scalar functions."""

    _no_grad: bool = False

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
            self.data: list[Scalar] = data.data
        elif isinstance(data, np.ndarray):
            self.data = [
                x if isinstance(x, Scalar) else Scalar(x, requires_grad=requires_grad)
                for x in data.flatten()
            ]
        elif isinstance(data, (list, tuple)):
            self.data = [
                x if isinstance(x, Scalar) else Scalar(x, requires_grad=requires_grad)
                for x in ag.utils.flatten(data)
            ]
        else:
            self.data = [
                data
                if isinstance(data, Scalar)
                else Scalar(data, requires_grad=requires_grad)
            ]
        if shape is None:
            self.shape: tuple[int, ...] = self._infer_shape(data)
        else:
            self.shape = tuple(shape)
        assert all(
            isinstance(dim, int) for dim in self.shape
        ), f"Invalid shape: {self.shape}"
        assert len(self.data) == math.prod(
            self.shape
        ), f"Shape {self.shape} does not match data length {len(self.data)}."
        if _child_nodes is None or self._no_grad:
            self._child_nodes: list[Tensor] = []
        else:
            assert isinstance(_child_nodes, list), "_child_nodes must be a list."
            assert all(
                isinstance(node, Tensor) for node in _child_nodes
            ), "_child_nodes must be a list of Tensors."
            self._child_nodes = _child_nodes
        self.requires_grad: bool = requires_grad
        self.name: Optional[str] = name

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
    def _infer_shape(x: AcceptedInput) -> tuple[int, ...]:
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
        elements: list[str] = []
        if self.shape == SCALAR_SHAPE:
            elements.append(f"value={self.data[0].data:.4f}")
        elements.append(f"shape={self.shape}")
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

    def _key_to_int_slices(self, key: Union[int, slice, tuple]) -> tuple:
        """Convert a key to a list of flattened indices."""
        # Normalize the index to a tuple of slices.
        if isinstance(key, int):
            insert_dims = []
            per_dim_slice = self._int_to_tuple(key)
        elif key is None:
            insert_dims = [0]
            per_dim_slice = tuple()
        elif isinstance(key, slice):
            insert_dims = []
            per_dim_slice = self._slice_to_tuple(key)
        elif isinstance(key, (tuple, list)):
            # Some values of `key` may be `None` if the user specified
            # `ag.newaxis`. Strip these out, and keep track of the dimensions that
            # were removed as we will need to insert singleton dimensions later.
            insert_dims: list[int] = [dim for dim, x in enumerate(key) if x is None]
            key = tuple(x for x in key if x is not None)
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
        if len(per_dim_slice) != len(self.shape):
            # If the user did not specify an index for each dimension, fill in
            # the missing dimensions with full slices.
            per_dim_slice += (slice(None, None, None),) * (
                len(self.shape) - len(per_dim_slice)
            )
        assert len(per_dim_slice) == len(self.shape), (
            f"Number of indices ({len(per_dim_slice)}) does not match number "
            f"of dimensions ({len(self.shape)})."
        )
        # Convert the slice to a list of slices.
        int_slices: list[slice] = [
            dim_slice.indices(dim_size)
            for dim_slice, dim_size in zip(per_dim_slice, self.shape)
        ]
        return int_slices, insert_dims

    def _int_slices_to_flattened_indices(self, int_slices: list[slice]) -> list[int]:
        """Convert a list of slices to a list of flattened indices."""
        # Convert the slices to ranges.
        int_ranges: list[range] = [
            range(start, stop, step) for start, stop, step in int_slices
        ]
        # Get the strides for each dimension.
        per_dim_strides: list[int] = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            per_dim_strides[i] = per_dim_strides[i + 1] * self.shape[i + 1]
        # Compute the flattened indices.
        flattened_indices = [
            self._nd_i_to_1d_i(nd_i) for nd_i in itertools.product(*int_ranges)
        ]
        return flattened_indices

    def __getitem__(self, key: Union[int, slice, tuple]) -> Tensor:
        """Return a subset of the tensor."""
        # Convert the key to a list of slices.
        int_slices, insert_dims = self._key_to_int_slices(key)
        # Convert the slices to a list of flattened indices.
        flattened_indices: list[int] = self._int_slices_to_flattened_indices(int_slices)
        sliced_data: list[Scalar] = []
        # Apply the slice to each dimension.
        for flattened_i in flattened_indices:
            sliced_data.append(self.data[flattened_i])
        # Compute the shape of the sliced tensor.
        sliced_shape_list: list[int] = [
            math.ceil((stop - start) / end)  # type: ignore
            for (start, stop, end) in int_slices
        ]
        # Remove any dimensions of size 1.
        sliced_shape: tuple[int, ...] = tuple(x for x in sliced_shape_list if x != 1)
        # Insert singleton dimensions.
        for dim in insert_dims:
            sliced_shape = sliced_shape[:dim] + (1,) + sliced_shape[dim:]
        # Create a new tensor with the sliced data.
        return Tensor(
            data=sliced_data,
            shape=sliced_shape,
            requires_grad=self.requires_grad,
            name=self.name,
        )

    def __setitem__(self, key: Union[int, slice, tuple], value: AcceptedInput) -> None:
        """Set a subset of the tensor."""
        # Convert the key to a list of slices.
        int_slices, _ = self._key_to_int_slices(key)
        # Convert the slices to a list of flattened indices.
        flattened_indices: list[int] = self._int_slices_to_flattened_indices(int_slices)
        # Convert the value to a tensor.
        value: Tensor = _to_tensor(value)
        # If the value is same size but different shape, broadcast it to the
        # shape of the slice.
        if value.size == self[key].size and value.shape != self[key].shape:
            value = value.reshape(self[key].shape)
        # Check that the shapes match.
        assert (
            value.shape == self[key].shape
        ), f"Value shape {value.shape} does not match tensor shape {self[key].shape}."
        # Update the data.
        for flattened_i, new_scalar in zip(flattened_indices, value.data):
            self.data[flattened_i] = new_scalar

    def broadcast_to_shape(self, shape: tuple[int, ...]) -> Tensor:
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

    def __gt__(self, other: AcceptedInput) -> Tensor:
        """Compare two tensors."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x > y for x, y in zip(l.data, r.data)],
            shape=l.shape,
            requires_grad=l.requires_grad or r.requires_grad,
            _child_nodes=[l, r],
        )

    def __iadd__(self, other: AcceptedInput) -> Tensor:
        """Add two tensors in-place."""
        l, r = _to_elementwise_op_tensor(self.identity(), other)
        self.data = [x + y for x, y in zip(l.data, r.data)]
        self.requires_grad = l.requires_grad or r.requires_grad
        self._child_nodes = [l, r]
        return self

    def __isub__(self, other: AcceptedInput) -> Tensor:
        """Subtract two tensors in-place."""
        l, r = _to_elementwise_op_tensor(self.identity(), other)
        self.data = [x - y for x, y in zip(l.data, r.data)]
        self.requires_grad = l.requires_grad or r.requires_grad
        self._child_nodes = [l, r]
        return self

    def __imul__(self, other: AcceptedInput) -> Tensor:
        """Multiply two tensors in-place."""
        l, r = _to_elementwise_op_tensor(self.identity(), other)
        self.data = [x * y for x, y in zip(l.data, r.data)]
        self.requires_grad = l.requires_grad or r.requires_grad
        self._child_nodes = [l, r]
        return self

    def __itruediv__(self, other: AcceptedInput) -> Tensor:
        """Divide two tensors in-place."""
        l, r = _to_elementwise_op_tensor(self.identity(), other)
        self.data = [x / y for x, y in zip(l.data, r.data)]
        self.requires_grad = l.requires_grad or r.requires_grad
        self._child_nodes = [l, r]
        return self

    def __ipow__(self, other: AcceptedInput) -> Tensor:
        """Raise a tensor to a power in-place."""
        l, r = _to_elementwise_op_tensor(self.identity(), other)
        self.data = [x**y for x, y in zip(l.data, r.data)]
        self.requires_grad = l.requires_grad or r.requires_grad
        self._child_nodes = [l, r]
        return self

    def identity(self) -> Tensor:
        """Return the tensor itself."""
        return Tensor(
            data=[x.identity() for x in self.data],
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
        r = _to_tensor(other)
        if l.size == 1 and r.size == 1:
            return l * r
        elif l.ndim == 1 and r.ndim == 1:
            return self._vector_dot(l, r)
        elif l.ndim == 2 and r.ndim == 1:
            assert l.shape[1] == r.shape[0]
            n_products: int = l.shape[0]
            out_shape: tuple[int, ...] = (n_products,)
            out_data: list[Scalar] = []
            for slice_i in range(n_products):
                dot_product: Scalar = self._vector_dot(l[slice_i, :], r)
                out_data.append(dot_product)
            return Tensor(
                data=out_data,
                shape=out_shape,
                requires_grad=l.requires_grad or r.requires_grad,
                _child_nodes=[l, r],
            )
        elif l.ndim == 1 and r.ndim == 2:
            assert l.shape[0] == r.shape[0]
            assert r.shape[1] == 1
            dot_product: Scalar = self._vector_dot(l, r.flatten())
            return self.from_scalar(dot_product)
        elif r.ndim >= 2:
            assert l.shape[-1] == r.shape[-2]
            out_shape = (*l.shape[:-1], r.shape[-1])
            if l.ndim == 1:
                l = l.reshape(1, 1, l.shape[-1])
            else:
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

    @property
    def is_scalar(self) -> bool:
        """Return whether the tensor is a scalar."""
        return self.shape == SCALAR_SHAPE

    @property
    def name(self) -> Optional[str]:
        """Return the name of the tensor."""
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        """Set the name of the tensor."""
        self._name = name
        # Set the names of the scalar values.
        if name is not None and self.shape == SCALAR_SHAPE:
            self.data[0].name = name
        elif name is not None and self.shape != SCALAR_SHAPE:
            ranges = [range(dim) for dim in self.shape]
            for nd_i in itertools.product(*ranges):
                flattened_i: int = self._nd_i_to_1d_i(nd_i)
                self.data[flattened_i].name = f"{name}{list(nd_i)}"

    def tile(self, reps: list[int] | tuple[int]) -> Tensor:
        """Tile the tensor."""
        assert isinstance(reps, (tuple, list)), "reps must be a list."
        assert all(isinstance(x, int) for x in reps), "reps must be a list of ints."
        assert len(reps) == len(self.shape), "The number of dimensions must match."
        result = self.identity()
        for axis in range(len(self.shape) - 1, -1, -1):
            if reps[axis] != 1:
                result = ag.concatenate([result] * reps[axis], axis=axis)
        return result

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

    def maximum(self, other: AcceptedInput) -> Tensor:
        """Compute the elementwise maximum of the tensor and another tensor."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x.maximum(y) for x, y in zip(l.data, r.data)],
            shape=self.shape,
            requires_grad=self.requires_grad or other.requires_grad,
            _child_nodes=[l, r],
        )

    def minimum(self, other: AcceptedInput) -> Tensor:
        """Compute the elementwise minimum of the tensor and another tensor."""
        l, r = _to_elementwise_op_tensor(self, other)
        return Tensor(
            data=[x.minimum(y) for x, y in zip(l.data, r.data)],
            shape=self.shape,
            requires_grad=self.requires_grad or other.requires_grad,
            _child_nodes=[l, r],
        )

    def clip(self, min_value: AcceptedInput, max_value: AcceptedInput) -> Tensor:
        """Clip the tensor to be between min and max."""
        return self.maximum(min_value).minimum(max_value)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function to the tensor."""
        return Tensor(
            data=[x.sigmoid() for x in self.data],
            shape=self.shape,
            requires_grad=self.requires_grad,
            _child_nodes=[self],
        )

    def log(self) -> Tensor:
        """Apply the log function to the tensor."""
        return Tensor(
            data=[x.log() for x in self.data],
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

    def exp(self) -> Tensor:
        """Apply the exp function to the tensor."""
        return Tensor(
            data=[x.exp() for x in self.data],
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

    def max(self, axis: Optional[int] = None) -> Tensor:
        """Compute the max of the tensor."""
        if axis is None:
            # Compute the max over all axes.
            return Tensor(
                data=ag.scalar.max(self.data),
                shape=SCALAR_SHAPE,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        elif isinstance(axis, int):
            assert 0 <= axis < len(self.shape), "axis out of bounds."
            new_shape = tuple(x for i, x in enumerate(self.shape) if i != axis)
            new_data = []
            for i in range(self.size // self.shape[axis]):
                new_data.append(
                    ag.scalar.max(
                        self.data[i * self.shape[axis] : (i + 1) * self.shape[axis]]
                    )
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
                    ag.scalar.max(
                        self.data[
                            i
                            * math.prod([self.shape[x] for x in axis]) : (i + 1)
                            * math.prod([self.shape[x] for x in axis])
                        ]
                    )
                )
            return Tensor(
                data=new_data,
                shape=new_shape,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        else:
            raise TypeError("axis must be an int or a list of ints.")

    def min(self, axis: Optional[int] = None) -> Tensor:
        """Compute the min of the tensor."""
        if axis is None:
            # Compute the min over all axes.
            return Tensor(
                data=ag.scalar.min(self.data),
                shape=SCALAR_SHAPE,
                requires_grad=self.requires_grad,
                _child_nodes=[self],
            )
        elif isinstance(axis, int):
            assert 0 <= axis < len(self.shape), "axis out of bounds."
            new_shape = tuple(x for i, x in enumerate(self.shape) if i != axis)
            new_data = []
            for i in range(self.size // self.shape[axis]):
                new_data.append(
                    ag.scalar.min(
                        self.data[i * self.shape[axis] : (i + 1) * self.shape[axis]]
                    )
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
                    ag.scalar.min(
                        self.data[
                            i
                            * math.prod([self.shape[x] for x in axis]) : (i + 1)
                            * math.prod([self.shape[x] for x in axis])
                        ]
                    )
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


class Parameter(Tensor):
    """A scalar value with a gradient that can be optimized."""

    def __init__(self, *args, **kwargs):
        """Initialize a scalar value with a gradient that can be optimized."""
        super().__init__(*args, **kwargs, requires_grad=True)  # type: ignore


def _to_tensor(x: AcceptedInput) -> Tensor:
    """Convert x to a Tensor."""
    return x if isinstance(x, Tensor) else Tensor(data=x)


def _tile_to_match_shape(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Tile x and y to match their shapes."""
    assert len(x.shape) == len(y.shape), "The number of dimensions must match."
    assert x.shape != y.shape, "The shapes must not match."
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


def _to_elementwise_op_tensor(
    x: AcceptedInput, y: AcceptedInput
) -> tuple[Tensor, Tensor]:
    """Ensure y is a Tensor, and broadcast x and y to the same shape."""
    x = _to_tensor(x)
    y = _to_tensor(y)
    if x.size == 1:
        # Handle scalar.
        x = x.reshape([1 for _ in y.shape])
    if y.size == 1:
        # Handle scalar.
        y = y.reshape([1 for _ in x.shape])
    if len(x.shape) < len(y.shape):
        # x has fewer dimensions than y. Try padding x with 1s.
        x = x.reshape([1 for _ in range(len(y.shape) - len(x.shape))] + list(x.shape))
    if len(y.shape) < len(x.shape):
        # y has fewer dimensions than x. Try padding y with 1s.
        y = y.reshape([1 for _ in range(len(x.shape) - len(y.shape))] + list(y.shape))
    assert len(x.shape) == len(y.shape), "The number of dimensions must match."
    return (x, y) if x.shape == y.shape else _tile_to_match_shape(x, y)
