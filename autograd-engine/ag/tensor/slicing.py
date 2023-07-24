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


def _int_to_slice(index: int, dim_size: int) -> slice:
    """Convert an integer index to a slice."""
    assert isinstance(index, int), "Index must be an integer."
    assert index < dim_size, "Index out of bounds."
    return slice(index, index + 1, 1)


def _int_to_tuple(index: int, shape: tuple[int, ...]) -> tuple:
    """Convert an integer index to a tuple."""
    assert isinstance(index, int), "Index must be an integer."
    assert index < shape[0], "Index out of bounds."
    # Index the first dimension, and return slice objects for the rest.
    return (_int_to_slice(index, dim=0),) + (slice(None, None, None),) * (
        len(shape) - 1
    )


def _tensor_to_tuple(index: "Tensor") -> tuple:
    """Convert a tensor index to a tuple."""
    assert index.dtype == int, "Index must be an integer tensor."
    assert index.is_scalar or index.ndim == 1, "Index must be a 1D tensor."
    return tuple(index)


def _slice_to_tuple(index: slice, shape: tuple[int, ...]) -> tuple:
    """Convert a slice index to a tuple."""
    assert isinstance(index, slice), "Index must be a slice."
    return (index,) + (slice(None, None, None),) * (len(shape) - 1)


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


def _key_to_int_slices(self, key: int | slice | tuple | "Tensor") -> tuple:
    """Convert a key to a list of flattened indices."""
    # Normalize the index to a tuple of slices.
    if isinstance(key, int):
        insert_dims = []
        per_dim_info = self._int_to_tuple(key)
    elif key is None:
        insert_dims = [0]
        per_dim_info = tuple()
    elif isinstance(key, ag.Tensor):
        insert_dims = []
        per_dim_info = self._tensor_to_tuple(key)
    elif isinstance(key, slice):
        insert_dims = []
        per_dim_info = self._slice_to_tuple(key)
    elif isinstance(key, (tuple, list)):
        # Some values of `key` may be `None` if the user specified
        # `ag.newaxis`. Strip these out, and keep track of the dimensions that
        # were removed as we will need to insert singleton dimensions later.
        insert_dims: list[int] = [dim for dim, x in enumerate(key) if x is None]
        key = tuple(x for x in key if x is not None)
        per_dim_info = tuple(
            self._int_to_slice(x, dim=dim) if isinstance(x, int) else x
            for dim, x in enumerate(key)
        )
    else:
        raise TypeError(f"Invalid index type {type(key)}.")
    # Help with type checking.
    assert isinstance(per_dim_info, tuple), f"Invalid index type {type(per_dim_info)}."
    assert all(
        isinstance(x, slice) for x in per_dim_info
    ), f"Invalid index type {per_dim_info}."
    if len(per_dim_info) != len(self.shape):
        # If the user did not specify an index for each dimension, fill in
        # the missing dimensions with full slices.
        per_dim_info += (slice(None, None, None),) * (
            len(self.shape) - len(per_dim_info)
        )
    assert len(per_dim_info) == len(self.shape), (
        f"Number of indices ({len(per_dim_info)}) does not match number "
        f"of dimensions ({len(self.shape)})."
    )

    # Convert the slice to a list of slices.
    int_slices: list[slice] = [
        dim_slice.indices(dim_size)
        for dim_slice, dim_size in zip(per_dim_info, self.shape)
    ]
    breakpoint()
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
    # Convert the slices to a list of flattened indices.
    flattened_indices: list[int] = self._int_slices_to_flattened_indices(int_slices)
    return flattened_indices, sliced_shape


def _int_slices_to_flattened_indices(
    int_slices: list[slice], shape: tuple[int, ...]
) -> list[int]:
    """Convert a list of slices to a list of flattened indices."""
    # Convert the slices to ranges.
    int_ranges: list[range] = [
        range(start, stop, step) for start, stop, step in int_slices
    ]
    # Get the strides for each dimension.
    per_dim_strides: list[int] = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        per_dim_strides[i] = per_dim_strides[i + 1] * shape[i + 1]
    # Compute the flattened indices.
    flattened_indices = [
        _nd_i_to_1d_i(nd_i) for nd_i in itertools.product(*int_ranges)
    ]
    return flattened_indices
