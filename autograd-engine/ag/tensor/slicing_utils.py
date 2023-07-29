"""Slicing utilities."""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class DimInfo:
    """Information about a dimension."""

    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]
    tensor: Optional["Tensor"]

    @classmethod
    def from_slice_and_dim_size(cls, s: slice, dim_size: int) -> DimInfo:
        """Create a `DimInfo` from a slice."""
        start, stop, step = s.indices(dim_size)
        return cls(
            start=start,
            stop=stop,
            step=step,
            tensor=None,
        )

    @classmethod
    def from_tensor(cls, t: Tensor) -> DimInfo:
        """Create a `DimInfo` from a tensor."""
        return cls(
            start=None,
            stop=None,
            step=None,
            tensor=t,
        )

    def __post_init__(self):
        """Post initialization."""
        if self.tensor is not None:
            # Avoid circular import.
            from ag.tensor.tensor import Tensor

            assert isinstance(self.tensor, Tensor)
            assert (
                self.tensor.ndim == 1
            ), f"Cannot index with a tensor of dimension {self.tensor.ndim}."
            assert self.start is None
            assert self.stop is None
            assert self.step is None
        else:
            assert isinstance(self.start, int)
            assert isinstance(self.stop, int)
            assert isinstance(self.step, int)

    @property
    def size(self) -> int:
        """Return the size of the dimension."""
        if self.tensor is not None:
            return len(self.tensor)
        else:
            return math.ceil((self.stop - self.start) / self.step)

    @property
    def indices(self) -> list[int]:
        """Return the indices of the dimension."""
        if self.tensor is not None:
            return [int(x.data) for x in self.tensor.data]
        else:
            return list(range(self.start, self.stop, self.step))


def _int_to_per_dim_info(index: int, shape: tuple[int, ...]) -> list[DimInfo]:
    """Convert an integer index to a tuple."""
    dim_size: int = shape[0]
    assert isinstance(index, int), "Index must be an integer."
    assert (
        index < dim_size
    ), f"Index {index} is out of bounds for dimension of size {dim_size}."
    return [DimInfo.from_slice_and_dim_size(slice(index, index + 1, 1), dim_size)]


def _tensor_to_per_dim_info(tensor_indices: "Tensor") -> list[DimInfo]:
    """Convert a tensor index to a tuple."""
    assert tensor_indices.dtype == int, "Index must be an integer tensor."
    # Remove all singleton dimensions.
    tensor_indices = tensor_indices.squeeze()
    assert tensor_indices.is_scalar or tensor_indices.ndim == 1, (
        f"Cannot index with a tensor of dimension {tensor_indices.ndim} and "
        f"shape {tensor_indices.shape}."
    )
    # Use tensor for 0th dim, and return slice objects for the rest.
    return [DimInfo.from_tensor(tensor_indices)]


def _slice_to_per_dim_info(s: slice, shape: tuple[int, ...]) -> list[DimInfo]:
    """Convert a slice index to a tuple."""
    dim_size: int = shape[0]
    assert isinstance(s, slice), "Index must be a slice."
    return [DimInfo.from_slice_and_dim_size(s, dim_size)]


def _nd_i_to_1d_i(shape: tuple[int, ...], nd_i: tuple[int, ...]) -> int:
    """Convert multi-dim indices to 1D index for `self.data`."""
    assert len(nd_i) == len(shape), (
        f"Number of indices ({len(nd_i)}) does not match number of "
        f"dimensions ({len(shape)})."
    )
    index: int = 0
    for dim, dim_i in enumerate(nd_i):
        assert dim_i < shape[dim], (
            f"Index {dim_i} is out of bounds for dimension {dim} with "
            f"shape {shape[dim]}."
        )
        index *= shape[dim]
        index += dim_i
    return index


def _per_dim_info_to_flattened_indices(
    per_dim_info: list[DimInfo], shape: tuple[int, ...]
) -> list[int]:
    """Convert a list of DimInfo to a list of flattened indices."""
    # Get the strides for each dimension.
    per_dim_strides: list[int] = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        per_dim_strides[i] = per_dim_strides[i + 1] * shape[i + 1]
    # Compute the flattened indices.
    per_dim_indices: list[list[int]] = [dim_info.indices for dim_info in per_dim_info]
    flattened_indices = [
        _nd_i_to_1d_i(shape=shape, nd_i=nd_i)
        for nd_i in itertools.product(*per_dim_indices)
    ]
    return flattened_indices


def _key_to_per_dim_info(
    key: int | slice | tuple | "Tensor",
    shape: tuple[int, ...],
    return_insert_dims: bool = True,
) -> list[DimInfo]:
    """Convert a key to list of per-dimension info."""
    # Avoid circular import.
    from ag.tensor.tensor import Tensor

    # Normalize the index to a tuple of slices.
    if isinstance(key, int):
        insert_dims = []
        per_dim_info = _int_to_per_dim_info(index=key, shape=shape)
    elif key is None:
        insert_dims = [0]
        per_dim_info = tuple()
    elif isinstance(key, Tensor):
        insert_dims = []
        per_dim_info = _tensor_to_per_dim_info(tensor_indices=key)
    elif isinstance(key, slice):
        insert_dims = []
        per_dim_info = _slice_to_per_dim_info(s=key, shape=shape)
    elif isinstance(key, (tuple, list)):
        # Some values of `key` may be `None` if the user specified
        # `ag.newaxis`. Strip these out, and keep track of the dimensions that
        # were removed as we will need to insert singleton dimensions later.
        insert_dims: list[int] = [dim for dim, x in enumerate(key) if x is None]
        per_dim_info = [
            _key_to_per_dim_info(key=x, shape=(dim_size,), return_insert_dims=False)[0]
            for x, dim_size in zip(key, shape)
            if x is not None
        ]
    else:
        raise TypeError(f"Invalid index type {type(key)}.")
    assert isinstance(per_dim_info, list), f"Invalid index type {type(per_dim_info)}."
    if len(per_dim_info) != len(shape):
        # If the user did not specify an index for each dimension, fill in
        # the missing dimensions with full slices.
        per_dim_info += [
            DimInfo.from_slice_and_dim_size(slice(None), dim_size)
            for dim_size in shape[len(per_dim_info) :]
        ]
    assert all(
        isinstance(x, DimInfo) for x in per_dim_info
    ), f"Invalid index type {per_dim_info}."
    if return_insert_dims:
        return per_dim_info, insert_dims
    else:
        return per_dim_info


def key_to_int_slices(
    key: int | slice | tuple | "Tensor", shape: tuple[int, ...]
) -> tuple:
    """Convert a key to a list of flattened indices."""
    per_dim_info, insert_dims = _key_to_per_dim_info(key=key, shape=shape)
    # Help with type checking.
    assert len(per_dim_info) == len(shape), (
        f"Number of indices ({len(per_dim_info)}) does not match number "
        f"of dimensions ({len(shape)})."
    )
    # Remove any dimensions of size 1.
    sliced_shape: tuple[int, ...] = tuple(
        dim_info.size for dim_info in per_dim_info if dim_info != 1
    )
    # Insert singleton dimensions.
    for dim in insert_dims:
        sliced_shape = sliced_shape[:dim] + (1,) + sliced_shape[dim:]
    # Convert the slices to a list of flattened indices.
    flattened_indices: list[int] = _per_dim_info_to_flattened_indices(
        per_dim_info=per_dim_info, shape=shape
    )
    return flattened_indices, sliced_shape
