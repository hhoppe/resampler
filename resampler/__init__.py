"""resampler: fast differentiable resizing and warping of arbitrary grids.

.. include:: ../README.md
"""


__docformat__ = 'google'
__version__ = '0.3.2'
__version_info__ = tuple(int(num) for num in __version__.split('.'))


import dataclasses
import functools
import math
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.ndimage
import scipy.sparse.linalg


try:
  import numba
except ModuleNotFoundError:
  pass


_DType = Any
_NDArray = Any  # To document np.ndarray[Any, Any] without enforcement.
_TensorflowTensor = Any  # To document tf.Tensor without enforcement.
_TorchTensor = Any  # To document torch.Tensor without enforcement.
_Array = Any  # To document any array class supported by _Arraylib.


def _check_eq(a: Any, b: Any) -> None:
  """If the two values or arrays are not equal, raise an exception with a useful message."""
  equal = np.all(a == b) if isinstance(a, np.ndarray) else a == b
  if not equal:
    raise AssertionError(f'{a!r} == {b!r}')


def _real_precision(dtype: _DType) -> _DType:
  """Return the type of the real part of a complex number."""
  return np.array([], dtype=dtype).real.dtype


def _get_precision(precision: Any, dtypes: List[_DType], weight_dtypes: List[_DType]) -> _DType:
  """Return dtype based on desired precision or on data and weight types."""
  precision = np.dtype(precision if precision is not None else
                       np.find_common_type([], [np.float32, *dtypes, *weight_dtypes]))
  if not np.issubdtype(precision, np.inexact):
    raise ValueError(f'Precision {precision} is not floating or complex.')
  check_complex = [precision, *dtypes]
  is_complex = [np.issubdtype(dtype, np.complexfloating) for dtype in check_complex]
  if len(set(is_complex)) != 1:
    raise ValueError(f'Types {",".join(str(dtype) for dtype in check_complex)}'
                     ' must be all real or all complex.')
  return precision


def _sinc(x: Any) -> _NDArray:
  """Return the value `np.sinc(x)` but improved to:
  (1) ignore underflow that occurs at 0.0 for np.float32, and
  (2) output exact zero for integer input values.
  """
  x = np.asarray(x)
  x_is_scalar = x.ndim == 0
  with np.errstate(under='ignore'):
    result = np.sinc(np.atleast_1d(x))
    result[x == np.floor(x)] = 0.0
    result[x == 0] = 1.0
    return result.item() if x_is_scalar else result


def _is_symmetric(matrix: Any, tol: float = 1e-10) -> bool:
  """Return True if the sparse matrix is symmetric."""
  norm: float = scipy.sparse.linalg.norm(matrix - matrix.T, np.inf)
  return norm <= tol


def _spline_filter1d(*args: Any, mode: Optional[str] = None, **kwargs: Any) -> _NDArray:
  """Wrapper to use 'grid-wrap' if scipy is a recent version."""
  if mode == 'wrap':
    scipy_version = tuple(int(num) for num in scipy.__version__.split('.'))
    mode = 'grid-wrap' if scipy_version >= (1, 6) else 'wrap'
  return scipy.ndimage.spline_filter1d(*args, mode=mode, **kwargs)


def _cache_sampled_1d_function(
    xmin: float, xmax: float, *, num_samples: int = 3_600, enable: bool = True,
) -> Callable[[Callable[[Any], _NDArray]], Callable[..., _NDArray]]:
  """Function decorator to linearly interpolate cached function values."""
  # Speed unchanged up to num_samples=12_000, then slow decrease until 100_000.

  def wrap_it(func: Callable[[Any], _NDArray]) -> Callable[..., _NDArray]:
    if not enable:
      @functools.wraps(func)
      def original_func(x: Any, *, mode: str = '') -> _NDArray:
        _check_eq(mode, '')
        return func(x)
      return original_func

    dx = (xmax - xmin) / num_samples
    x = np.linspace(xmin, xmax + dx, num_samples + 2, dtype=np.float32)
    samples_func = func(x)
    assert np.all(samples_func[[0, -1, -2]] == 0.0)

    @functools.wraps(func)
    def interpolate_using_cached_samples(x: Any, *, mode: str = 'linear') -> _NDArray:
      if mode == 'exact':
        return func(x)
      index_float = np.clip((x - xmin) / dx, 0.0, num_samples)
      index = index_float.astype(np.int64)
      if mode == 'nearest':
        return samples_func[index]
      if mode == 'linear':
        frac = np.subtract(index_float, index, dtype=np.float32)
        return (1 - frac) * samples_func[index] + frac * samples_func[index + 1]
      raise ValueError(f'Mode {mode} is not one of: exact, nearest, linear.')

    return interpolate_using_cached_samples

  return wrap_it


class _DownsampleIn2dUsingBoxFilter:
  """Fast 2D box-filter downsampling using cached numba-jitted functions."""

  def __init__(self) -> None:
    # Downsampling function for params (dtype, block_height, block_width, ch).
    self._jitted_function: Dict[Tuple[_DType, int, int, int], Callable[[_NDArray], _NDArray]] = {}

  def __call__(self, array: _NDArray, shape: Tuple[int, int]) -> _NDArray:
    assert 'numba' in globals()
    assert array.ndim in (2, 3), array.ndim
    _check_eq(len(shape), 2)
    dtype = array.dtype
    a = array[..., None] if array.ndim == 2 else array
    height, width, ch = a.shape
    new_height, new_width = shape
    if height % new_height != 0 or width % new_width != 0:
      raise ValueError(f'Shape {array.shape} not a multiple of {shape}.')
    block_height, block_width = height // new_height, width // new_width

    def func(array: _NDArray) -> _NDArray:  # pylint: disable=too-many-branches
      new_height = array.shape[0] // block_height
      new_width = array.shape[1] // block_width
      result = np.empty((new_height, new_width, ch), dtype)
      totals = np.empty(ch, dtype)
      scale = dtype.type(1.0 / (block_height * block_width))
      for y in range(new_height):  # pylint: disable=too-many-nested-blocks
        for x in range(new_width):
          # y2, x2 = y * block_height, x * block_width ?
          if ch == 1:  # All the branches involve compile-time constants.
            total = dtype.type(0.0)
            for yy in range(block_height):
              for xx in range(block_width):
                total += array[y * block_height + yy, x * block_width + xx, 0]
            result[y, x, 0] = total * scale
          elif ch == 3:
            total0 = total1 = total2 = dtype.type(0.0)
            for yy in range(block_height):
              for xx in range(block_width):
                # pixel = array[y2 + yy, x2 + xx] ?
                total0 += array[y * block_height + yy, x * block_width + xx, 0]
                total1 += array[y * block_height + yy, x * block_width + xx, 1]
                total2 += array[y * block_height + yy, x * block_width + xx, 2]
            result[y, x, 0] = total0 * scale
            result[y, x, 1] = total1 * scale
            result[y, x, 2] = total2 * scale
          elif block_height * block_width >= 9:
            for c in range(ch):
              totals[c] = 0.0
            for yy in range(block_height):
              for xx in range(block_width):
                for c in range(ch):
                  totals[c] += array[y * block_height + yy, x * block_width + xx, c]
            for c in range(ch):
              result[y, x, c] = totals[c] * scale
          else:
            for c in range(ch):
              total = dtype.type(0.0)
              for yy in range(block_height):
                for xx in range(block_width):
                  total += array[y * block_height + yy, x * block_width + xx, c]
              result[y, x, c] = total * scale
      return result

    signature = dtype, block_height, block_width, ch
    jitted_function = self._jitted_function.get(signature)
    if not jitted_function:
      if 0:
        print(f'Creating numba jit-wrapper for {signature}.')
      jitted_function = self._jitted_function[signature] = numba.njit(func)
    result = jitted_function(a)
    return result[..., 0] if array.ndim == 2 else result


_downsample_in_2d_using_box_filter = _DownsampleIn2dUsingBoxFilter()


@dataclasses.dataclass
class _Arraylib:
  """Abstract base class for abstraction of array libraries."""

  arraylib: str
  """Name of array library (e.g., 'numpy', 'tensorflow', 'torch')."""

  array: _Array

  @staticmethod
  def recognize(array: Any) -> bool:
    """Return True if `array` is recognized by this array library."""

  def numpy(self) -> _NDArray:
    """Return a `numpy` version of this array."""

  def dtype(self) -> _DType:
    """Return the numpy dtype of the array."""

  def astype(self, dtype: Any) -> _Array:
    """Return the array with values cast as numpy dtype."""

  def clip(self, low: Any, high: Any, dtype: Any = None) -> _Array:
    """Return the array with values clipped to the range [low, high]."""

  def square(self) -> _Array:
    """Return a new array with values squared."""

  def sqrt(self) -> _Array:
    """Return a new array with values square-rooted."""

  def gather(self, table: Any) -> _Array:
    """Return `table[array]`."""

  def where(self, if_true: Any, if_false: Any) -> _Array:
    """Return `where(array, if_true, if_false)`."""

  def transpose(self, axes: Sequence[int]) -> _Array:
    """Return the array with its axes permuted."""

  def concatenate(self, arrays: Sequence[_Array], axis: int) -> _Array:
    """Return the concatenation of arrays; uses self only to determine arraylib!"""


class _NumpyArraylib(_Arraylib):
  """Numpy implementation of the array abstraction."""

  def __init__(self, array: Any) -> None:
    super().__init__(arraylib='numpy', array=np.asarray(array))

  @staticmethod
  def recognize(array: Any) -> bool:
    return isinstance(array, np.ndarray)

  def numpy(self) -> _NDArray:
    return self.array

  def dtype(self) -> _DType:
    return self.array.dtype

  def astype(self, dtype: Any) -> _Array:
    return self.array.astype(dtype, copy=False)

  def clip(self, low: Any, high: Any, dtype: Any = None) -> _Array:
    return self.array.clip(low, high, dtype=dtype)

  def square(self) -> _Array:
    return np.square(self.array)

  def sqrt(self) -> _Array:
    return np.sqrt(self.array)

  def gather(self, table: Any) -> _Array:
    indices = self.array
    return table[indices]

  def where(self, if_true: Any, if_false: Any) -> _Array:
    condition = self.array
    return np.where(condition, if_true, if_false)

  def transpose(self, axes: Sequence[int]) -> _Array:
    return np.transpose(self.array, tuple(axes))

  def concatenate(self, arrays: Sequence[_Array], axis: int) -> _Array:
    return np.concatenate(arrays, axis)


class _TensorflowArraylib(_Arraylib):
  """Tensorflow implementation of the array abstraction."""

  def __init__(self, array: Any) -> None:
    import tensorflow as tf
    super().__init__(arraylib='tensorflow', array=tf.convert_to_tensor(array))

  @staticmethod
  def recognize(array: Any) -> bool:
    # Eager: tensorflow.python.framework.ops.Tensor
    # Non-eager: tensorflow.python.ops.resource_variable_ops.ResourceVariable
    return type(array).__module__.startswith('tensorflow.')

  def numpy(self) -> _NDArray:
    return self.array.numpy()

  def dtype(self) -> _DType:
    return np.dtype(self.array.dtype.as_numpy_dtype)

  def astype(self, dtype: Any) -> _Array:
    import tensorflow as tf
    return tf.cast(self.array, dtype)

  def clip(self, low: Any, high: Any, dtype: Any = None) -> _Array:
    import tensorflow as tf
    array = self.array
    if dtype is not None:
      array = tf.cast(array, dtype)
    return tf.clip_by_value(array, low, high)

  def square(self) -> _Array:
    import tensorflow as tf
    return tf.square(self.array)

  def sqrt(self) -> _Array:
    import tensorflow as tf
    return tf.sqrt(self.array)

  def gather(self, table: Any) -> _Array:
    import tensorflow as tf
    indices = self.array
    if _arr_dtype(indices) in (np.uint8, np.uint16):
      indices = tf.cast(indices, np.int32)
    return tf.gather(table, indices, axis=None)

  def where(self, if_true: Any, if_false: Any) -> _Array:
    import tensorflow as tf
    condition = self.array
    return tf.where(condition, if_true, if_false)

  def transpose(self, axes: Sequence[int]) -> _Array:
    import tensorflow as tf
    return tf.transpose(self.array, perm=tuple(axes))

  def concatenate(self, arrays: Sequence[_Array], axis: int) -> _Array:
    import tensorflow as tf
    return tf.concat(arrays, axis)


class _TorchArraylib(_Arraylib):
  """Torch implementation of the array abstraction."""

  def __init__(self, array: Any) -> None:
    import torch
    super().__init__(arraylib='torch', array=torch.as_tensor(array))

  @staticmethod
  def recognize(array: Any) -> bool:
    return type(array).__module__ == 'torch'

  def numpy(self) -> _NDArray:
    return self.array.numpy()

  def dtype(self) -> _DType:
    import torch
    numpy_type = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
        torch.uint8: np.uint8,  # No uint16, uint32, uint64.
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }[self.array.dtype]
    return np.dtype(numpy_type)

  def astype(self, dtype: Any) -> _Array:
    import torch
    torch_type = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.uint8: torch.uint8,  # No uint16, uint32, uint64.
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }[np.dtype(dtype).type]
    return self.array.type(torch_type)

  def clip(self, low: Any, high: Any, dtype: Any = None) -> _Array:
    array = self.array
    array = _arr_astype(array, dtype) if dtype is not None else array
    return array.clip(low, high)

  def square(self) -> _Array:
    return self.array.square()

  def sqrt(self) -> _Array:
    return self.array.sqrt()

  def gather(self, table: Any) -> _Array:
    import torch
    indices = self.array
    return torch.as_tensor(table)[indices.type(torch.int64)]

  def where(self, if_true: Any, if_false: Any) -> _Array:
    condition = self.array
    return if_true.where(condition, if_false)

  def transpose(self, axes: Sequence[int]) -> _Array:
    import torch
    return torch.permute(self.array, tuple(axes))

  def concatenate(self, arrays: Sequence[_Array], axis: int) -> _Array:
    import torch
    return torch.cat(tuple(arrays), axis)


def _as_arr(array: _Array) -> _Arraylib:
  """Return array wrapped as an `_Arraylib` for dispatch of functions."""
  for cls in _DICT_ARRAYLIBS.values():
    if cls.recognize(array):
      return cls(array)
  raise AssertionError(
      f'{array} {type(array)} {type(array).__module__} unrecognized by {ARRAYLIBS}.')


def _arr_arraylib(array: _Array) -> str:
  """Return the name of the arraylib representing the array."""
  return _as_arr(array).arraylib


def _arr_numpy(array: _Array) -> _NDArray:
  """Return a `numpy` version of this array."""
  return _as_arr(array).numpy()


def _arr_dtype(array: _Array) -> _DType:
  """Return the numpy dtype of the array."""
  return _as_arr(array).dtype()


def _arr_astype(array: _Array, dtype: Any) -> _Array:
  """Return the array with values cast as numpy dtype."""
  return _as_arr(array).astype(dtype)


def _arr_clip(array: _Array, low: Any, high: Any, dtype: Any = None) -> _Array:
  """Return the array with values clipped to the range [low, high]."""
  return _as_arr(array).clip(low, high, dtype)


def _arr_square(array: _Array) -> _Array:
  """Return a new array with values squared."""
  return _as_arr(array).square()


def _arr_sqrt(array: _Array) -> _Array:
  """Return a new array with values square-rooted."""
  return _as_arr(array).sqrt()


def _arr_gather(table: Any, indices: Any) -> _Array:
  """Return `table[indices]`."""
  return _as_arr(indices).gather(table)


def _arr_where(condition: Any, if_true: Any, if_false: Any) -> _Array:
  """Return `where(condition, if_true, if_false)`."""
  return _as_arr(condition).where(if_true, if_false)


def _arr_transpose(array: _Array, axes: Sequence[int]) -> _Array:
  """Return the array with its axes permuted."""
  return _as_arr(array).transpose(axes)


def _arr_concatenate(arrays: Sequence[Any], axis: int) -> _Array:
  """Return the concatenation of arrays."""
  return _as_arr(arrays[0]).concatenate(arrays, axis)


def _arr_swapaxes(array: _Array, axis0: int, axis1: int) -> _Array:
  """Return the array with its axes swapped."""
  ndim = array.ndim
  assert 0 <= axis0 < ndim and 0 <= axis1 < ndim, (axis0, axis1, ndim)
  axes = list(range(ndim))
  axes[axis0] = axis1
  axes[axis1] = axis0
  return _arr_transpose(array, axes)


def _make_array(array: Any, arraylib: str) -> _Array:
  """Create an array defined in the library `arraylib`."""
  return _DICT_ARRAYLIBS[arraylib](array).array


_DICT_ARRAYLIBS = {
    'numpy': _NumpyArraylib,
    'tensorflow': _TensorflowArraylib,
    'torch': _TorchArraylib,
}

ARRAYLIBS = list(_DICT_ARRAYLIBS)
"""Supported array libraries."""


def _block_shape_with_min_size(shape: Tuple[int, ...], min_size: int,
                               compact: bool = True) -> Tuple[int, ...]:
  """Return shape of block (of size at least `min_size`) to subdivide shape."""
  if np.prod(shape) < min_size:
    raise ValueError(f'Shape {shape} smaller than min_size {min_size}.')
  if compact:
    root = int(math.ceil(min_size**(1 / len(shape))))
    block_shape = np.minimum(shape, root)
    for dim in range(len(shape)):
      if block_shape[dim] == 2 and block_shape.prod() >= min_size * 2:
        block_shape[dim] = 1
    for dim in range(len(shape) - 1, -1, -1):
      if block_shape.prod() < min_size:
        block_shape[dim] = shape[dim]
  else:
    block_shape = np.ones_like(shape)
    for dim in range(len(shape) - 1, -1, -1):
      if block_shape.prod() < min_size:
        block_shape[dim] = min(shape[dim], math.ceil(min_size / block_shape.prod()))
  return tuple(block_shape)


def _array_split(array: _Array, axis: int, num_sections: int) -> List[Any]:
  """Split `array` into `num_sections` along `axis`."""
  assert 0 <= axis < array.ndim
  assert 1 <= num_sections <= array.shape[axis]

  if 0:
    split = np.array_split(array, num_sections, axis=axis)  # Numpy-specific.

  else:
    # Adapted from https://github.com/numpy/numpy/blob/main/numpy/lib/shape_base.py#L739-L792.
    num_total = array.shape[axis]
    num_each, num_extra = divmod(num_total, num_sections)
    section_sizes = [0] + num_extra * [num_each + 1] + (num_sections - num_extra) * [num_each]
    div_points = np.array(section_sizes).cumsum()
    split = []
    tmp = _arr_swapaxes(array, axis, 0)
    for i in range(num_sections):
      split.append(_arr_swapaxes(tmp[div_points[i]:div_points[i + 1]], axis, 0))

  return split


def _split_array_into_blocks(array: _Array, block_shape: Sequence[int],
                             start_axis: int = 0) -> Any:
  """Split an array into nested lists of blocks of size at most block_shape."""
  # See https://stackoverflow.com/a/50305924.  (If the block_shape is known to
  # exactly partition the array, see https://stackoverflow.com/a/16858283.)
  if len(block_shape) > array.ndim:
    raise ValueError(f'Block ndim {len(block_shape)} > array ndim {array.ndim}.')
  if start_axis == len(block_shape):
    return array

  num_sections = math.ceil(array.shape[start_axis] / block_shape[start_axis])
  split = _array_split(array, start_axis, num_sections)
  return [_split_array_into_blocks(split_a, block_shape, start_axis + 1) for split_a in split]


def _map_function_over_blocks(blocks: Any, func: Callable[[Any], Any]) -> Any:
  """Apply `func` to each block in the nested lists of blocks."""
  if isinstance(blocks, list):
    return [_map_function_over_blocks(block, func) for block in blocks]
  return func(blocks)


def _merge_array_from_blocks(blocks: Any, axis: int = 0) -> _NDArray:
  """Merge an array from nested lists of array blocks."""
  # More general than np.block() because the blocks can have additional dims.
  if isinstance(blocks, list):
    new_blocks = [_merge_array_from_blocks(block, axis + 1) for block in blocks]
    return _arr_concatenate(new_blocks, axis)
  return blocks


@dataclasses.dataclass
class Gridtype:
  """Abstract base class for grid-types such as 'dual' and 'primal'.

  In resampling operations, the grid-type may be specified separately as `src_gridtype` for the
  source domain and `dst_gridtype` for the destination domain.  Moreover, the grid-type may be
  specified per domain dimension.

  Examples:
    `resize(source, shape, gridtype='primal')`  # Sets both src and dst.

    `resize(source, shape, src_gridtype=['dual', 'primal'],
           dst_gridtype='dual')`  # Source is dual in dim0 and primal in dim1.
  """

  name: str
  """Gridtype name."""

  def min_size(self) -> int:
    """Return the necessary minimum number of grid samples."""

  def size_in_samples(self, size: int) -> int:
    """Return the size of the domain in units of inter-sample spacing."""

  def point_from_index(self, index: Any, size: int) -> _NDArray:
    """Return [0.0, 1.0] coordinates given [0, size - 1] indices."""

  def index_from_point(self, point: Any, size: int) -> _NDArray:
    """Return location x given coordinates [0.0, 1.0], where x == 0.0 is the
    first grid sample and x == size - 1.0 is the last grid sample."""

  def reflect(self, index: Any, size: int) -> _NDArray:
    """Map integer sample indices to interior ones using boundary reflection."""

  def wrap(self, index: Any, size: int) -> _NDArray:
    """Map integer sample indices to interior ones using wrapping."""

  def reflect_clamp(self, index: Any, size: int) -> _NDArray:
    """Map integer sample indices to interior ones using reflect-clamp."""


class DualGridtype(Gridtype):
  """Samples are at the center of cells in a uniform partition of the domain.

  For a unit-domain dimension with N samples, each sample 0 <= i <= N has position (i + 0.5) / N,
  e.g., [0.125, 0.375, 0.625, 0.875] for N = 4.
  """

  def __init__(self) -> None:
    super().__init__(name='dual')

  def min_size(self) -> int:
    return 1

  def size_in_samples(self, size: int) -> int:
    return size

  def point_from_index(self, index: Any, size: int) -> _NDArray:
    return (index + 0.5) / size

  def index_from_point(self, point: Any, size: int) -> _NDArray:
    return point * size - 0.5

  def reflect(self, index: Any, size: int) -> _NDArray:
    index = np.mod(index, size * 2)
    return np.where(index < size, index, 2 * size - 1 - index)

  def wrap(self, index: Any, size: int) -> _NDArray:
    return np.mod(index, size)

  def reflect_clamp(self, index: Any, size: int) -> _NDArray:
    return np.minimum(np.where(index < 0, -1 - index, index), size - 1)


class PrimalGridtype(Gridtype):
  """Samples are at the vertices of cells in a uniform partition of the domain.

  For a unit-domain dimension with N samples, each sample 0 <= i <= N has position i / (N - 1),
  e.g., [0, 1/3, 2/3, 1] for N = 4.
  """

  def __init__(self) -> None:
    super().__init__(name='primal')

  def min_size(self) -> int:
    return 2

  def size_in_samples(self, size: int) -> int:
    return size - 1

  def point_from_index(self, index: Any, size: int) -> _NDArray:
    return index / (size - 1)

  def index_from_point(self, point: Any, size: int) -> _NDArray:
    return point * (size - 1)

  def reflect(self, index: Any, size: int) -> _NDArray:
    index = np.mod(index, size * 2 - 2)
    return np.where(index < size, index, 2 * size - 2 - index)

  def wrap(self, index: Any, size: int) -> _NDArray:
    return np.mod(index, size - 1)

  def reflect_clamp(self, index: Any, size: int) -> _NDArray:
    return np.minimum(np.abs(index), size - 1)


_DICT_GRIDTYPES = {
    'dual': DualGridtype(),
    'primal': PrimalGridtype(),
}

GRIDTYPES = list(_DICT_GRIDTYPES)
"""Shortcut names for some predefined grid types, specified per dimension.
These are: `'dual'` and `'primal'`.""";

def _get_gridtype(gridtype: Union[str, Gridtype]) -> Gridtype:
  """Return a `Gridtype`, which can be specified as a name in `GRIDTYPES`."""
  return gridtype if isinstance(gridtype, Gridtype) else _DICT_GRIDTYPES[gridtype]


def _get_gridtypes(
    gridtype: Union[None, str, Gridtype],
    src_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]],
    dst_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]],
    src_ndim: int, dst_ndim: int) -> Tuple[List[Gridtype], List[Gridtype]]:
  """Return per-dim source and destination grid types given all parameters."""
  if gridtype is None and src_gridtype is None and dst_gridtype is None:
    gridtype = 'dual'
  if gridtype is not None:
    if src_gridtype is not None:
      raise ValueError('Cannot have both gridtype and src_gridtype.')
    if dst_gridtype is not None:
      raise ValueError('Cannot have both gridtype and dst_gridtype.')
    src_gridtype = dst_gridtype = gridtype
  src_gridtype2 = [_get_gridtype(g) for g in np.broadcast_to(np.array(src_gridtype), src_ndim)]
  dst_gridtype2 = [_get_gridtype(g) for g in np.broadcast_to(np.array(dst_gridtype), dst_ndim)]
  return src_gridtype2, dst_gridtype2


class RemapCoordinates:
  """Abstract base class for modifying the specified coordinates prior to evaluating the
  reconstruction kernels."""

  def __call__(self, point: _NDArray) -> _NDArray:
    pass


class NoRemapCoordinates(RemapCoordinates):
  """The coordinates are not remapped."""

  def __call__(self, point: _NDArray) -> _NDArray:
    return point


class MirrorRemapCoordinates(RemapCoordinates):
  """The coordinates are reflected across the domain boundaries so that they lie in the unit
  interval.  The resulting function is continuous but not smooth across the boundaries."""

  def __call__(self, point: _NDArray) -> _NDArray:
    point = np.mod(point, 2.0)
    return np.where(point >= 1.0, 2.0 - point, point)


class TileRemapCoordinates(RemapCoordinates):
  """The coordinates are mapped to the unit interval using a "modulo 1.0" operation.  The resulting
  function is generally discontinuous across the domain boundaries."""

  def __call__(self, point: _NDArray) -> _NDArray:
    return np.mod(point, 1.0)


@dataclasses.dataclass
class ExtendSamples:
  """Abstract base class for replacing references to grid samples exterior to the unit domain by
  affine combinations of interior sample(s) and possibly the constant value (cval)."""

  uses_cval: bool = False
  """True if some exterior samples are defined in terms of `cval`, i.e., if the computed weight
  is non-affine."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    """Detect references to exterior samples, i.e., entries of `index` that lie outside the
    interval [0, size), and update these indices (and possibly their associated weights) to
    reference only interior samples.  Return `new_index, new_weight`."""


class ReflectExtendSamples(ExtendSamples):
  """Find the interior sample by reflecting across domain boundaries."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    index = gridtype.reflect(index, size)
    return index, weight


class WrapExtendSamples(ExtendSamples):
  """Wrap the interior samples periodically.  For a 'primal' grid, the last
  sample is ignored as its value is replaced by the first sample."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    index = gridtype.wrap(index, size)
    return index, weight


class ClampExtendSamples(ExtendSamples):
  """Use the nearest interior sample."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    index = index.clip(0, size - 1)
    return index, weight


class ReflectClampExtendSamples(ExtendSamples):
  """Extend the grid samples from [0, 1] into [-1, 0] using reflection and then define grid
  samples outside [-1, 1] as that of the nearest sample."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    index = gridtype.reflect_clamp(index, size)
    return index, weight


class BorderExtendSamples(ExtendSamples):
  """Let all exterior samples have the constant value (`cval`)."""

  def __init__(self) -> None:
    super().__init__(uses_cval=True)

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    low = index < 0
    weight[low] = 0.0
    index[low] = 0
    high = index >= size
    weight[high] = 0.0
    index[high] = size - 1
    return index, weight


class ValidExtendSamples(ExtendSamples):
  """Assign all domain samples weight 1 and all outside samples weight 0.
  Compute a weighted reconstruction and divide by the reconstructed weight."""

  def __init__(self) -> None:
    super().__init__(uses_cval=True)

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    low = index < 0
    weight[low] = 0.0
    index[low] = 0
    high = index >= size
    weight[high] = 0.0
    index[high] = size - 1
    sum_weight = weight.sum(axis=-1)
    nonzero_sum = sum_weight != 0.0
    np.divide(weight, sum_weight[..., None], out=weight, where=nonzero_sum[..., None])
    return index, weight


class LinearExtendSamples(ExtendSamples):
  """Linearly extrapolate beyond boundary samples."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    if size < 2:
      index = gridtype.reflect(index, size)
      return index, weight
    # For each boundary, define new columns in index and weight arrays to represent the last and
    # next-to-last samples.  When we later construct the sparse resize matrix, we will sum the
    # duplicate index entries.
    low = index < 0
    high = index >= size
    w = np.empty((*weight.shape[:-1], weight.shape[-1] + 4), dtype=weight.dtype)
    x = index
    w[..., -4] = ((1 - x) * weight).sum(where=low, axis=-1)
    w[..., -3] = ((x) * weight).sum(where=low, axis=-1)
    x = (size - 1) - index
    w[..., -2] = ((x) * weight).sum(where=high, axis=-1)
    w[..., -1] = ((1 - x) * weight).sum(where=high, axis=-1)
    weight[low] = 0.0
    index[low] = 0
    weight[high] = 0.0
    index[high] = size - 1
    w[..., :-4] = weight
    weight = w
    new_index = np.empty(w.shape, dtype=index.dtype)
    new_index[..., :-4] = index
    # Let matrix (including zero values) be banded.
    new_index[..., -4:] = np.where(w[..., -4:] != 0.0, [0, 1, size - 2, size - 1], index[..., :1])
    index = new_index
    return index, weight


class QuadraticExtendSamples(ExtendSamples):
  """Quadratically extrapolate beyond boundary samples."""

  def __call__(self, index: _NDArray, weight: _NDArray, size: int,
               gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    # [Keys 1981] suggests this as x[-1] = 3*x[0] - 3*x[1] + x[2], calling it "cubic precision",
    # but it seems just quadratic.
    if size < 3:
      index = gridtype.reflect(index, size)
      return index, weight
    low = index < 0
    high = index >= size
    w = np.empty((*weight.shape[:-1], weight.shape[-1] + 6), dtype=weight.dtype)
    x = index
    w[..., -6] = (((0.5 * x - 1.5) * x + 1) * weight).sum(where=low, axis=-1)
    w[..., -5] = (((-x + 2) * x) * weight).sum(where=low, axis=-1)
    w[..., -4] = (((0.5 * x - 0.5) * x) * weight).sum(where=low, axis=-1)
    x = (size - 1) - index
    w[..., -3] = (((0.5 * x - 0.5) * x) * weight).sum(where=high, axis=-1)
    w[..., -2] = (((-x + 2) * x) * weight).sum(where=high, axis=-1)
    w[..., -1] = (((0.5 * x - 1.5) * x + 1) * weight).sum(where=high, axis=-1)
    weight[low] = 0.0
    index[low] = 0
    weight[high] = 0.0
    index[high] = size - 1
    w[..., :-6] = weight
    weight = w
    new_index = np.empty(w.shape, dtype=index.dtype)
    new_index[..., :-6] = index
    # Let matrix (including zero values) be banded.
    new_index[..., -6:] = np.where(
        w[..., -6:] != 0.0, [0, 1, 2, size - 3, size - 2, size - 1], index[..., :1])
    index = new_index
    return index, weight


@dataclasses.dataclass
class OverrideExteriorValue:
  """Abstract base class to set the value outside some domain extent to a
  constant value (`cval`)."""

  boundary_antialiasing: bool = True
  """Antialias the pixel values adjacent to the boundary of the extent."""

  uses_cval: bool = False
  """Modify some weights to introduce references to `cval` constant value."""

  def __call__(self, weight: _NDArray, point: _NDArray) -> None:
    """For all `point` outside some extent, modify the weight to be zero."""

  def override_using_signed_distance(self, weight: _NDArray, point: _NDArray,
                                     signed_distance: _NDArray) -> None:
    """Reduce sample weights for "outside" values based on the signed distance function,
    to effectively assign the constant value `cval`."""
    all_points_inside_domain = np.all(signed_distance <= 0.0)
    if all_points_inside_domain:
      return
    if self.boundary_antialiasing and min(point.shape) >= 2:
      # For discontinuous coordinate mappings, we may need to somehow ignore
      # the large finite differences computed across the map discontinuities.
      gradient = np.gradient(point)
      gradient_norm = np.linalg.norm(np.atleast_2d(gradient), axis=0)
      signed_distance_in_samples = signed_distance / (gradient_norm + 1e-20)
      # Opacity is in linear space, which is correct if Gamma is set.
      opacity = (0.5 - signed_distance_in_samples).clip(0.0, 1.0)
      weight *= opacity[..., None]
    else:
      is_outside = signed_distance > 0.0
      weight[is_outside, :] = 0.0


class NoOverrideExteriorValue(OverrideExteriorValue):
  """The function value is not overridden."""

  def __call__(self, weight: _NDArray, point: _NDArray) -> None:
    pass


class UnitDomainOverrideExteriorValue(OverrideExteriorValue):
  """Values outside the unit interval [0, 1] are replaced by constant `cval`."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(uses_cval=True, **kwargs)

  def __call__(self, weight: _NDArray, point: _NDArray) -> None:
    signed_distance = abs(point - 0.5) - 0.5  # Boundaries at 0.0 and 1.0.
    self.override_using_signed_distance(weight, point, signed_distance)


class PlusMinusOneOverrideExteriorValue(OverrideExteriorValue):
  """Values outside the interval [-1, 1] are replaced by the constant `cval`."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(uses_cval=True, **kwargs)

  def __call__(self, weight: _NDArray, point: _NDArray) -> None:
    signed_distance = abs(point) - 1.0  # Boundaries at -1.0 and 1.0.
    self.override_using_signed_distance(weight, point, signed_distance)


@dataclasses.dataclass
class Boundary:
  """Domain boundary rules.  These define the reconstruction over the source domain near and beyond
  the domain boundaries.  The rules may be specified separately for each domain dimension."""

  name: str = ''
  """Boundary rule name."""

  coord_remap: RemapCoordinates = NoRemapCoordinates()
  """Modify specified coordinates prior to evaluating the reconstruction kernels."""

  extend_samples: ExtendSamples = ReflectExtendSamples()
  """Define the value of each grid sample outside the unit domain as an affine combination of
  interior sample(s) and possibly the constant value (cval)."""

  override_value: OverrideExteriorValue = NoOverrideExteriorValue()
  """Set the value outside some extent to a constant value (cval)."""

  uses_cval: bool = dataclasses.field(init=False)
  """True if weights may be non-affine, involving the constant value (cval)."""

  def __post_init__(self) -> None:
    self.uses_cval = self.extend_samples.uses_cval or self.override_value.uses_cval

  def preprocess_coordinates(self, point: _NDArray) -> _NDArray:
    """Modify coordinates prior to evaluating the filter kernels."""
    # Antialiasing across the tile boundaries may be feasible but seems hard.
    point = self.coord_remap(point)
    return point

  def apply(self, index: _NDArray, weight: _NDArray, point: _NDArray,
            size: int, gridtype: Gridtype) -> Tuple[_NDArray, _NDArray]:
    """Replace exterior samples by combinations of interior samples."""
    index, weight = self.extend_samples(index, weight, size, gridtype)
    self.override_reconstruction(weight, point)
    return index, weight

  def override_reconstruction(self, weight: _NDArray, point: _NDArray) -> None:
    """For points outside an extent, modify weight to zero to assign `cval`."""
    self.override_value(weight, point)


_DICT_BOUNDARIES = {
    'reflect':  # (a.k.a. reflected, symm, symmetric, mirror, grid-mirror.)
        Boundary('reflect', extend_samples=ReflectExtendSamples()),
    'wrap':  # (a.k.a. periodic, repeat, grid-wrap.)
        Boundary('wrap', extend_samples=WrapExtendSamples()),
    'tile':  # (a.k.a. wrap.)
        Boundary('title', coord_remap=TileRemapCoordinates(),
                 extend_samples=ReflectExtendSamples()),
    'clamp':  # (a.k.a. clamped, nearest, edge, clamp-to-edge.)
        Boundary('clamp', extend_samples=ClampExtendSamples()),
    'border':  # (a.k.a. grid-constant.)
        Boundary('border', extend_samples=BorderExtendSamples()),
    'natural':  # (a.k.a. IgnoreOutside, renormalize.)
        Boundary('natural', extend_samples=ValidExtendSamples(),
                 override_value=UnitDomainOverrideExteriorValue()),
    'linear_constant':
        Boundary('linear_constant', extend_samples=LinearExtendSamples(),
                 override_value=UnitDomainOverrideExteriorValue()),
    'quadratic_constant':
        Boundary('quadratic_constant', extend_samples=QuadraticExtendSamples(),
                 override_value=UnitDomainOverrideExteriorValue()),
    'reflect_clamp':  # (a.k.a. mirror-clamp-to-edge)
        Boundary('reflect_clamp', extend_samples=ReflectClampExtendSamples()),
    'constant':  # (a.k.a. constant, reflect-constant.)
        Boundary('constant', extend_samples=ReflectExtendSamples(),
                 override_value=UnitDomainOverrideExteriorValue()),
    'linear':
        Boundary('linear', extend_samples=LinearExtendSamples()),
    'quadratic':
        Boundary('quadratic', extend_samples=QuadraticExtendSamples()),
}

BOUNDARIES = list(_DICT_BOUNDARIES)
"""Shortcut names for some predefined boundary rules.
These include: `'reflect'`, `'wrap'`, `'tile'`, `'clamp'`, `'border'`, etc.
See the source code for extensibility."""

_OFTUSED_BOUNDARIES = ('reflect wrap tile clamp border natural'
                       ' linear_constant quadratic_constant'.split())
"""A useful subset of `BOUNDARIES` for visualization in figures."""

def _get_boundary(boundary: Union[str, Boundary]) -> Boundary:
  """Return a `Boundary`, which can be specified as a name in `BOUNDARIES`."""
  return boundary if isinstance(boundary, Boundary) else _DICT_BOUNDARIES[boundary]


@dataclasses.dataclass
class Filter:
  """Abstract base class for filter kernel functions.

  Each kernel is assumed to be a zero-phase filter, i.e., to be symmetric in a support
  interval [-radius, radius].  (Some sites instead define kernels over the interval [0, N]
  where N = 2 * radius.)

  Portions of this code are adapted from the C++ library in
  https://github.com/hhoppe/Mesh-processing-library/blob/master/libHh/Filter.cpp

  See also http://hhoppe.com/proj/filtering/.
  """

  name: str
  """Filter kernel name."""

  radius: float
  """Max absolute value of x for which self(x) is nonzero."""

  interpolating: bool = True
  """True if self(0) == 1.0 and self(i) == 0.0 for all nonzero integers i."""

  continuous: bool = True
  """True if the kernel function has C^0 continuity."""

  partition_of_unity: bool = True
  """True if the convolution of the kernel with a Dirac comb reproduces the
  unity function."""

  unit_integral: bool = True
  """True if the integral of the kernel function is 1."""

  requires_digital_filter: bool = False
  """True if the filter needs a pre/post digital filter for interpolation."""

  def __call__(self, x: Any) -> _NDArray:
    """Return evaluation of filter kernel at locations x."""


class ImpulseFilter(Filter):
  """See https://en.wikipedia.org/wiki/Dirac_delta_function."""

  def __init__(self) -> None:
    super().__init__(name='impulse', radius=1e-20, continuous=False, partition_of_unity=False)

  def __call__(self, x: Any) -> _NDArray:
    raise AssertionError('The Impulse is infinitely narrow, so cannot be directly evaluated.')


class BoxFilter(Filter):
  """See https://en.wikipedia.org/wiki/Box_function.

  The kernel function has value 1.0 over the half-open interval [-.5, .5).
  """

  def __init__(self) -> None:
    super().__init__(name='box', radius=0.5, continuous=False)

  def __call__(self, x: Any) -> _NDArray:
    return np.where((-0.5 <= x) & (x < 0.5), 1.0, 0.0)
    # An alternative definition is symmetric with values 0.5 at the endpoints:
    # x = np.abs(x)
    # return np.where(x < 0.5, 1.0, np.where(x == 0.5, 0.5, 0.0))


class TrapezoidFilter(Filter):
  """Filter for antialiased "area-based" filtering.

  Args:
    radius: Specifies the support [-radius, radius] of the filter, where 0.5 < radius <= 1.0.
      The special case `radius = None` is a placeholder that indicates that the filter will be
      replaced by a trapezoid of the appropriate radius (based on scaling) for correct
      antialiasing in both minification and magnification.

  This filter is similar to the BoxFilter but with linearly sloped sides.  It has value 1.0
  in the interval abs(x) <= 1.0 - radius and decreases linearly to value 0.0 in the interval
  1.0 - radius <= abs(x) <= radius, always with value 0.5 at x = 0.5.
  """

  def __init__(self, *, radius: Optional[float] = None) -> None:
    if radius is None:
      super().__init__(name='trapezoid', radius=0.0)
      return
    if not 0.5 < radius <= 1.0:
      raise ValueError(f'Radius {radius} is outside the range (0.5, 1.0].')
    super().__init__(name=f'trapezoid_{radius}', radius=radius)

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    assert 0.5 < self.radius <= 1.0
    return ((0.5 + 0.25 / (self.radius - 0.5)) - (0.5 / (self.radius - 0.5)) * x).clip(0.0, 1.0)


class TriangleFilter(Filter):
  """See https://en.wikipedia.org/wiki/Triangle_function.

  Also known as the hat or tent function.  It is used for piecewise-linear
  (or bilinear, or trilinear, ...) interpolation.
  """

  def __init__(self) -> None:
    super().__init__(name='triangle', radius=1.0)

  def __call__(self, x: Any) -> _NDArray:
    return (1.0 - np.abs(x)).clip(0.0, 1.0)


class CubicFilter(Filter):
  """Family of cubic filters parameterized by two scalar parameters.

  Args:
    b: first scalar parameter.
    c: second scalar parameter.

  See https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters and
  https://doi.org/10.1145/378456.378514.

  [D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer graphics.
  Computer Graphics (Proceedings of ACM SIGGRAPH 1988), 22(4):221-228, 1988.]

  - The filter has quadratic precision iff b + 2 * c == 1.
  - The filter is interpolating iff b == 0.
  - (b=1, c=0) is the (non-interpolating) cubic B-spline basis;
  - (b=1/3, c=1/3) is the Mitchell filter;
  - (b=0, c=0.5) is the Catmull-Rom spline (which has cubic precision);
  - (b=0, c=0.75) is the "sharper cubic" used in Photoshop and OpenCV.
  """

  def __init__(self, *, b: float, c: float, name: Optional[str] = None) -> None:
    name = f'cubic_b{b}_c{c}' if name is None else name
    super().__init__(name=name, radius=2.0, interpolating=(b == 0))
    self.b, self.c = b, c

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    b, c = self.b, self.c
    f3, f2, f0 = 2 - 9/6*b - c, -3 + 2*b + c, 1 - 1/3*b
    g3, g2, g1, g0 = -b/6 - c, b + 5*c, -2*b - 8*c, 8/6*b + 4*c
    # (np.polynomial.polynomial.polyval(x, [f0, 0, f2, f3]) is almost
    # twice as slow; see also https://stackoverflow.com/questions/24065904)
    v01 = ((f3 * x + f2) * x) * x + f0
    v12 = ((g3 * x + g2) * x + g1) * x + g0
    return np.where(x < 1.0, v01, np.where(x < 2.0, v12, 0.0))


class CatmullRomFilter(CubicFilter):
  """Cubic filter with cubic precision.  Also known as Keys filter.

  [E. Catmull, R. Rom.  A class of local interpolating splines.  Computer aided geometric
  design, 1974]
  [Wikipedia](https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull%E2%80%93Rom_spline)

  [R. G. Keys.  Cubic convolution interpolation for digital image processing.
  IEEE Trans. on Acoustics, Speech, and Signal Processing, 29(6), 1981.]
  https://ieeexplore.ieee.org/document/1163711/.
  """

  def __init__(self) -> None:
    super().__init__(b=0, c=0.5, name='cubic')


class MitchellFilter(CubicFilter):
  """See https://doi.org/10.1145/378456.378514.

  [D. P. Mitchell and A. N. Netravali.  Reconstruction filters in computer graphics.  Computer
  Graphics (Proceedings of ACM SIGGRAPH 1988), 22(4):221-228, 1988.]
  """

  def __init__(self) -> None:
    super().__init__(b=1/3, c=1/3, name='mitchell')


class SharpCubicFilter(CubicFilter):
  """Cubic filter that is sharper than Catmull-Rom filter.

  Used by some tools including OpenCV and Photoshop.

  See https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters and
  http://entropymine.com/resamplescope/notes/photoshop/.
  """

  def __init__(self) -> None:
    super().__init__(b=0, c=0.75, name='sharpcubic')


class LanczosFilter(Filter):
  """High-quality filter: sinc function modulated by a sinc window.

  Args:
    radius: Specifies the support window [-radius, radius] over which the filter is nonzero.
    sampled: If True, use a discretized approximation for improved speed.

  See https://en.wikipedia.org/wiki/Lanczos_kernel.
  """

  def __init__(self, *, radius: int, sampled: bool = True) -> None:
    super().__init__(name=f'lanczos_{radius}', radius=radius, partition_of_unity=False,
                     unit_integral=False)

    @_cache_sampled_1d_function(xmin=-radius, xmax=radius, enable=sampled)
    def _eval(x: Any) -> _NDArray:
      x = np.abs(x)
      # Note that window[n] = sinc(2*n/N - 1), with 0 <= n <= N.
      # But, x = n - N/2, or equivalently, n = x + N/2, with -N/2 <= x <= N/2.
      window = _sinc(x / radius)  # Zero-phase function w_0(x).
      return np.where(x < radius, _sinc(x) * window, 0.0)

    self.function = _eval

  def __call__(self, x: Any) -> _NDArray:
    return self.function(x)


class GeneralizedHammingFilter(Filter):
  """Sinc function modulated by a Hamming window.

  Args:
    radius: Specifies the support window [-radius, radius] over which the filter is nonzero.
    a0: Scalar parameter, where 0.0 < a0 < 1.0.  The case of a0=0.5 is the Hann filter.

  See https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows,
  and hamming() in https://github.com/scipy/scipy/blob/master/scipy/signal/windows/windows.py.

  Generalized version of np.hamming() and np.hanning().
  """

  def __init__(self, *, radius: int, a0: float) -> None:
    super().__init__(
        name=f'hamming_{radius}',
        radius=radius,
        partition_of_unity=False,  # 1:1.00242  av=1.00188  sd=0.00052909
        unit_integral=False,       # 1.00188
    )
    assert 0.0 < a0 < 1.0
    self.a0 = a0

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    # Note that window[n] = a0 - (1 - a0) * cos(2 * pi * n / N), 0 <= n <= N.
    # With n = x + N/2, we get the zero-phase function w_0(x):
    window = self.a0 + (1.0 - self.a0) * np.cos(np.pi / self.radius * x)
    return np.where(x < self.radius, _sinc(x) * window, 0.0)


class KaiserFilter(Filter):
  """Sinc function modulated by a Kaiser-Bessel window.

  See https://en.wikipedia.org/wiki/Kaiser_window, and example use in:
  [Karras et al. 20201.  Alias-free generative adversarial networks.
  https://arxiv.org/pdf/2106.12423.pdf].

  Use np.kaiser()?

  Args:
    radius: Value L/2 in the definition.  It may be fractional for a (digital) resizing filter
      (sample spacing s != 1) with an even number of samples (dual grid), e.g., Eq. (6)
      in [Karras et al. 2021] --- this effects the precise shape of the window function.
    beta: Determines the trade-off between main-lobe width and side-lobe level.
    sampled: If True, use a discretized approximation for improved speed.
  """

  def __init__(self, *, radius: float, beta: float, sampled: bool = True) -> None:
    assert beta >= 0.0
    super().__init__(name=f'kaiser_{radius}_{beta}', radius=radius, partition_of_unity=False,
                     unit_integral=False)

    @_cache_sampled_1d_function(xmin=-math.ceil(radius), xmax=math.ceil(radius), enable=sampled)
    def _eval(x: Any) -> _NDArray:
      x = np.abs(x)
      window = np.i0(beta * np.sqrt((1.0 - np.square(x / radius)).clip(0.0, 1.0))) / np.i0(beta)
      return np.where(x <= radius + 1e-6, _sinc(x) * window, 0.0)

    self.function = _eval

  def __call__(self, x: Any) -> _NDArray:
    return self.function(x)


class BsplineFilter(Filter):
  """B-spline of a non-negative degree.

  Args:
    degree: The polynomial degree of the B-spline segments.
      - With `degree=0`, it is like `BoxFilter` except with f(0.5) = f(-0.5) = 0.
      - With `degree=1`, it is identical to `TriangleFilter`.
      - With `degree >= 2`, it is no longer interpolating.

  See [Carl de Boor.  A practical guide to splines.  Springer, 2001.]
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
  """

  def __init__(self, *, degree: int) -> None:
    if degree < 0:
      raise ValueError(f'Bspline of degree {degree} is invalid.')
    radius = (degree + 1) / 2
    super().__init__(name=f'bspline{degree}', radius=radius, interpolating=(degree <= 1))
    t = list(range(degree + 2))
    self.bspline = scipy.interpolate.BSpline.basis_element(t)

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    return np.where(x < self.radius, self.bspline(x + self.radius), 0.0)


class CardinalBsplineFilter(Filter):
  """Interpolating B-spline, achieved with aid of digital pre or post filter.

  Args:
    degree: The polynomial degree of the B-spline segments.
    sampled: If True, use a discretized approximation for improved speed.

  See [Hou and Andrews.  Cubic splines for image interpolation and digital filtering, 1978] and
  [Unser et al.  Fast B-spline transforms for continuous image representation and interpolation,
  1991].
  """

  def __init__(self, *, degree: int, sampled: bool = True) -> None:
    self.degree = degree
    if degree < 0:
      raise ValueError(f'Bspline of degree {degree} is invalid.')
    radius = (degree + 1) / 2
    super().__init__(name=f'cardinal{degree}', radius=radius, requires_digital_filter=True)
    t = list(range(degree + 2))
    bspline = scipy.interpolate.BSpline.basis_element(t)

    @_cache_sampled_1d_function(xmin=-radius, xmax=radius, enable=sampled)
    def _eval(x: Any) -> _NDArray:
      x = np.abs(x)
      return np.where(x < radius, bspline(x + radius), 0.0)

    self.function = _eval

  def __call__(self, x: Any) -> _NDArray:
    return self.function(x)


class OmomsFilter(Filter):
  """OMOMS interpolating filter, with aid of digital pre or post filter.

  Args:
    degree: The polynomial degree of the filter segments.

  Optimal MOMS (maximal-order-minimal-support) function; see [Blu and Thevenaz, MOMS: Maximal-order
  interpolation of minimal support, 2001].
  https://infoscience.epfl.ch/record/63074/files/blu0101.pdf
  """

  def __init__(self, *, degree: int) -> None:
    if degree not in (3, 5):
      raise ValueError(f'Degree {degree} not supported.')
    super().__init__(name=f'omoms{degree}', radius=(degree + 1) / 2, requires_digital_filter=True)
    self.degree = degree

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    if self.degree == 3:
      v01 = ((0.5 * x - 1.0) * x + 3/42) * x + 26/42
      v12 = ((-7/42 * x + 1.0) * x - 85/42) * x + 58/42
      return np.where(x < 1.0, v01, np.where(x < 2.0, v12, 0.0))
    if self.degree == 5:
      v01 = ((((-1/12 * x + 1/4) * x - 5/99) * x - 9/22) * x - 1/792) * x + 229/440
      v12 = ((((1/24 * x - 3/8) * x + 505/396) * x - 83/44) * x + 1351/1584) * x + 839/2640
      v23 = ((((-1/120 * x + 1/8) * x - 299/396) * x + 101/44) * x - 27811/7920) * x + 5707/2640
      return np.where(x < 1.0, v01, np.where(x < 2.0, v12, np.where(x < 3.0, v23, 0.0)))
    raise AssertionError


class GaussianFilter(Filter):
  r"""See https://en.wikipedia.org/wiki/Gaussian_function.

  Args:
    standard_deviation: Sets the Gaussian $\sigma$.  The default value is 1.25/3.0, which
      creates a kernel that is as-close-as-possible to a partition of unity.
  """

  DEFAULT_STANDARD_DEVIATION = 1.25 / 3.0
  """This value creates a kernel that is as-close-as-possible to a partition of unity; see
  mesh_processing/test/GridOp_test.cpp: `0.93503:1.06497     av=1           sd=0.0459424`.
  Another possibility is 0.5, as suggested on p. 4 of [Ken Turkowski.  Filters for common
  resampling tasks, 1990] for kernels with a support of 3 pixels.
  https://cadxfem.org/inf/ResamplingFilters.pdf
  """

  def __init__(self, *, standard_deviation: float = DEFAULT_STANDARD_DEVIATION) -> None:
    super().__init__(name=f'gaussian_{standard_deviation:.3f}',
                     radius=np.ceil(8.0 * standard_deviation),  # Sufficiently large.
                     interpolating=False,
                     partition_of_unity=False)
    self.standard_deviation = standard_deviation

  def __call__(self, x: Any) -> _NDArray:
    x = np.abs(x)
    sdv = self.standard_deviation
    v0r = np.exp(np.square(x / sdv) / -2.0) / (np.sqrt(math.tau) * sdv)
    return np.where(x < self.radius, v0r, 0.0)


class NarrowBoxFilter(Filter):
  """Compact footprint, used for visualization of grid sample location.

  Args:
    radius: Specifies the support [-radius, radius] of the narrow box function.  (The default
      value 0.199 is an inexact 0.2 to avoid numerical ambiguities.)
  """

  def __init__(self, *, radius: float = 0.199) -> None:
    super().__init__(name='narrowbox', radius=radius, continuous=False, unit_integral=False,
                     partition_of_unity=False)

  def __call__(self, x: Any) -> _NDArray:
    radius = self.radius
    magnitude = 1.0
    return np.where((-radius <= x) & (x < radius), magnitude, 0.0)


_DICT_FILTERS = {
    'impulse': ImpulseFilter(),  # a.k.a. 'nearest'
    'box': BoxFilter(),  # non-antialiased box, e.g. ImageMagick box.
    'trapezoid': TrapezoidFilter(),  # "area" antialiasing, e.g., cv.INTER_AREA
    'triangle': TriangleFilter(),  # a.k.a. 'linear'  ('bilinear' in 2D)
    'cubic': CatmullRomFilter(),  # a.k.a. 'catmullrom', 'keys', 'bicubic'.
    'sharpcubic': SharpCubicFilter(),  # cv.INTER_CUBIC, torch 'bicubic'.
    'lanczos3': LanczosFilter(radius=3),
    'lanczos5': LanczosFilter(radius=5),  # optionally: sampled=False
    'lanczos10': LanczosFilter(radius=10),
    'cardinal3': CardinalBsplineFilter(degree=3),
    'cardinal5': CardinalBsplineFilter(degree=5),
    'omoms3': OmomsFilter(degree=3),
    'omoms5': OmomsFilter(degree=5),
    'hamming3': GeneralizedHammingFilter(radius=3, a0=25/46),  # (a0 = ~0.54)
    'kaiser3': KaiserFilter(radius=3.0, beta=7.12),
    'gaussian': GaussianFilter(),
    'bspline3': BsplineFilter(degree=3),
    'mitchell': MitchellFilter(),  # a.k.a. 'mitchellcubic'
    'narrowbox': NarrowBoxFilter(),
    # Not in FILTERS:
    'hann3': GeneralizedHammingFilter(radius=3, a0=0.5),
    'lanczos4': LanczosFilter(radius=4),
}

FILTERS = [filter for filter in _DICT_FILTERS if filter not in 'hann3 lanczos4'.split()]
"""Shortcut names for some predefined filter kernels.
These names expand to:
```
{
    'impulse': ImpulseFilter(),  # a.k.a. 'nearest'
    'box': BoxFilter(),  # non-antialiased box, e.g. ImageMagick box.
    'trapezoid': TrapezoidFilter(),  # "area" antialiasing, e.g., cv.INTER_AREA
    'triangle': TriangleFilter(),  # a.k.a. 'linear'  ('bilinear' in 2D)
    'cubic': CatmullRomFilter(),  # a.k.a. 'catmullrom', 'keys', 'bicubic'.
    'sharpcubic': SharpCubicFilter(),  # cv.INTER_CUBIC, torch 'bicubic'.
    'lanczos3': LanczosFilter(radius=3),
    'lanczos5': LanczosFilter(radius=5),  # optionally: sampled=False
    'lanczos10': LanczosFilter(radius=10),
    'cardinal3': CardinalBsplineFilter(degree=3),
    'cardinal5': CardinalBsplineFilter(degree=5),
    'omoms3': OmomsFilter(degree=3),
    'omoms5': OmomsFilter(degree=5),
    'hamming3': GeneralizedHammingFilter(radius=3, a0=25/46),  # (a0 = ~0.54)
    'kaiser3': KaiserFilter(radius=3.0, beta=7.12),
    'gaussian': GaussianFilter(),
    'bspline3': BsplineFilter(degree=3),
    'mitchell': MitchellFilter(),  # a.k.a. 'mitchellcubic'
    'narrowbox': NarrowBoxFilter(),
}
```"""  # formatting??

def _get_filter(filter: Union[str, Filter]) -> Filter:
  """Return a `Filter`, which can be specified as a name string key in `FILTERS`."""
  return filter if isinstance(filter, Filter) else _DICT_FILTERS[filter]


def _to_float_01(array: _Array, dtype: Any) -> _Array:
  """Scale uint to the range [0.0, 1.0], and clip float to [0.0, 1.0]."""
  array_dtype = _arr_dtype(array)
  dtype = np.dtype(dtype)
  assert np.issubdtype(dtype, np.floating)
  if array_dtype in (np.uint8, np.uint16, np.uint32):
    if _arr_arraylib(array) == 'numpy':
      return np.multiply(array, 1 / np.iinfo(array_dtype).max, dtype=dtype)
    return _arr_astype(array, dtype) / np.iinfo(array_dtype).max
  assert np.issubdtype(array_dtype, np.floating)
  return _arr_clip(array, 0.0, 1.0, dtype=dtype)


def _from_float(array: _Array, dtype: Any) -> _Array:
  """Convert a float in range [0.0, 1.0] to uint or float type."""
  assert np.issubdtype(_arr_dtype(array), np.floating)
  dtype = np.dtype(dtype)
  if dtype.type in (np.uint8, np.uint16):
    return _arr_astype(array * np.float32(np.iinfo(dtype).max) + 0.5, dtype)
  if dtype.type == np.uint32:
    return _arr_astype(array * np.float64(np.iinfo(dtype).max) + 0.5, dtype)
  assert np.issubdtype(dtype, np.floating)
  return _arr_astype(array, dtype)


@dataclasses.dataclass
class Gamma:
  """Abstract base class for transfer functions on sample values.

  Image/video content is often stored using a color component transfer function.
  See https://en.wikipedia.org/wiki/Gamma_correction.

  Converts between integer types and [0.0, 1.0] internal value range.
  """

  name: str
  """Name of component transfer function."""

  def decode(self, array: _Array, dtype: Any = np.float32) -> _Array:
    """Decode source sample values into floating-point, possibly nonlinearly.

    Uint source values are mapped to the range [0.0, 1.0].
    """

  def encode(self, array: _Array, dtype: Any) -> _Array:
    """Encode float signal into destination samples, possibly nonlinearly.

    Uint destination values are mapped from the range [0.0, 1.0].
    """


class IdentityGamma(Gamma):
  """Identity component transfer function."""

  def __init__(self) -> None:
    super().__init__('identity')

  def decode(self, array: _Array, dtype: Any = np.float32) -> _Array:
    dtype = np.dtype(dtype)
    assert np.issubdtype(dtype, np.inexact)
    if np.issubdtype(_arr_dtype(array), np.unsignedinteger):
      return _to_float_01(array, dtype)
    return _arr_astype(array, dtype)

  def encode(self, array: _Array, dtype: Any) -> _Array:
    dtype = np.dtype(dtype)
    assert np.issubdtype(dtype, np.number)
    if np.issubdtype(dtype, np.unsignedinteger):
      return _from_float(_arr_clip(array, 0.0, 1.0), dtype)
    if np.issubdtype(dtype, np.integer):
      return _arr_astype(array + 0.5, dtype)
    return _arr_astype(array, dtype)


class PowerGamma(Gamma):
  """Gamma correction using a power function."""

  def __init__(self, power: float) -> None:
    super().__init__(name=f'power_{power}')
    self.power = power

  def decode(self, array: _Array, dtype: Any = np.float32) -> _Array:
    dtype = np.dtype(dtype)
    assert np.issubdtype(dtype, np.floating)
    if _arr_dtype(array) == np.uint8 and self.power != 2:
      decode_table = self.decode(np.arange(256, dtype=dtype) / 255)
      return _arr_gather(decode_table, array)

    array = _to_float_01(array, dtype)
    return _arr_square(array) if self.power == 2 else array**self.power

  def encode(self, array: _Array, dtype: Any) -> _Array:
    array = _arr_clip(array, 0.0, 1.0)
    array = _arr_sqrt(array) if self.power == 2 else array**(1.0 / self.power)
    return _from_float(array, dtype)


class SrgbGamma(Gamma):
  """Gamma correction using sRGB; see https://en.wikipedia.org/wiki/SRGB."""

  def __init__(self) -> None:
    super().__init__(name='srgb')

  def decode(self, array: _Array, dtype: Any = np.float32) -> _Array:
    dtype = np.dtype(dtype)
    assert np.issubdtype(dtype, np.floating)
    if _arr_dtype(array) == np.uint8:
      decode_table = self.decode(np.arange(256, dtype=dtype) / 255)
      return _arr_gather(decode_table, array)

    x = _to_float_01(array, dtype)
    return _arr_where(x > 0.04045, ((x + 0.055) / 1.055)**2.4, x / 12.92)

  def encode(self, array: _Array, dtype: Any) -> _Array:
    x = _arr_clip(array, 0.0, 1.0)
    # Unfortunately, exponentiation is slow, and np.digitize() is even slower.
    x = _arr_where(x > 0.0031308, x**(1.0 / 2.4) * 1.055 - (0.055 - 1e-17), x * 12.92)
    return _from_float(x, dtype)


_DICT_GAMMAS = {
    'identity': IdentityGamma(),
    'power2': PowerGamma(2.0),
    'power22': PowerGamma(2.2),
    'srgb': SrgbGamma(),
}

GAMMAS = list(_DICT_GAMMAS)
"""Shortcut names for some predefined gamma-correction schemes.
These are: `'identity'`, `'power2'`, `'power22'`, and `'srgb'`.""";

def _get_gamma(gamma: Union[str, Gamma]) -> Gamma:
  """Return a `Gamma`, which can be specified as a name in `GAMMAS`."""
  return gamma if isinstance(gamma, Gamma) else _DICT_GAMMAS[gamma]


def _get_src_dst_gamma(gamma: Union[None, str, Gamma],
                       src_gamma: Union[None, str, Gamma], dst_gamma: Union[None, str, Gamma],
                       src_dtype: _DType, dst_dtype: _DType) -> Tuple[Gamma, Gamma]:
  if gamma is None and src_gamma is None and dst_gamma is None:
    src_uint = np.issubdtype(src_dtype, np.unsignedinteger)
    dst_uint = np.issubdtype(dst_dtype, np.unsignedinteger)
    if src_uint and dst_uint:
      # The default might ideally be 'srgb' but that conversion is costlier.
      gamma = 'power2'
    elif not src_uint and not dst_uint:
      gamma = 'identity'
    else:
      raise ValueError('Gamma must be specified given that source is'
                       f' {src_dtype} and destination is {dst_dtype}.')
  if gamma is not None:
    if src_gamma is not None:
      raise ValueError('Cannot specify both gamma and src_gamma.')
    if dst_gamma is not None:
      raise ValueError('Cannot specify both gamma and dst_gamma.')
    src_gamma = dst_gamma = gamma
  assert src_gamma and dst_gamma
  src_gamma = _get_gamma(src_gamma)
  dst_gamma = _get_gamma(dst_gamma)
  return src_gamma, dst_gamma


def _create_resize_matrix(      # pylint: disable=too-many-statements
    src_size: int,
    dst_size: int,
    src_gridtype: Gridtype,
    dst_gridtype: Gridtype,
    boundary: Boundary,
    filter: Filter,
    prefilter: Optional[Filter] = None,
    scale: float = 1.0,
    translate: float = 0.0,
    dtype: _DType = np.float64,
    arraylib: str = 'numpy') -> Tuple[Any, Any]:
  """Compute affine weights for 1D resampling from `src_size` to `dst_size`.

  Compute a sparse matrix in which each row expresses a destination sample value as a combination
  of source sample values depending on the boundary rule.  If the combination is non-affine,
  the remainder (returned as `cval_weight`) is the contribution of the special constant value
  (cval) defined outside the domain.

  Args:
    src_size: The number of samples within the source 1D domain.
    dst_size: The number of samples within the destination 1D domain.
    src_gridtype: Placement of the samples in the source domain grid.
    dst_gridtype: Placement of the output samples in the destination domain grid.
    boundary: The reconstruction boundary rule.
    filter: The reconstruction kernel (used for upsampling/magnification).
    prefilter: The prefilter kernel (used for downsampling/minification).  If it is `None`,
      `filter` is used.
    scale: Scaling factor applied when mapping the source domain onto the destination domain.
    translate: Offset applied when mapping the scaled source domain onto the destination domain.
    dtype: Precision of computed resize matrix entries.
    arraylib: Representation of output.  Must be an element of `ARRAYLIBS`.

  Returns:
    sparse_matrix: Matrix whose rows express output sample values as affine combinations of the
      source sample values.
    cval_weight: Optional vector expressing the additional contribution of the constant value
      (`cval`) to the combination in each row of `sparse_matrix`.  It equals one minus the sum of
      the weights in each matrix row.
  """
  if src_size < src_gridtype.min_size():
    raise ValueError(f'Source size {src_size} is too small for resize.')
  if dst_size < dst_gridtype.min_size():
    raise ValueError(f'Destination size {dst_size} is too small for resize.')
  prefilter = filter if prefilter is None else prefilter
  dtype = np.dtype(dtype)
  assert np.issubdtype(dtype, np.floating)

  scaling = dst_gridtype.size_in_samples(dst_size) / src_gridtype.size_in_samples(src_size) * scale
  is_minification = scaling < 1.0
  filter = prefilter if is_minification else filter
  if filter.name == 'trapezoid':
    filter = TrapezoidFilter(radius=(0.5 + 0.5 * min(scaling, 1.0 / scaling)))
  radius = filter.radius
  num_samples = int(np.ceil(radius * 2 / scaling) if is_minification else np.ceil(radius * 2))

  dst_index = np.arange(dst_size, dtype=dtype)
  # Destination sample locations in unit domain [0, 1].
  dst_position = dst_gridtype.point_from_index(dst_index, dst_size)

  src_position = (dst_position - translate) / scale
  src_position = boundary.preprocess_coordinates(src_position)

  # Sample positions mapped back to source unit domain [0, 1].
  src_float_index = src_gridtype.index_from_point(src_position, src_size)
  src_first_index = np.floor(src_float_index + (0.5 if num_samples % 2 == 1 else 0.0)
                             ).astype(np.int32) - (num_samples - 1) // 2

  sample_index = np.arange(num_samples, dtype=np.int32)
  src_index = src_first_index[:, None] + sample_index  # (dst_size, num_samples)

  def get_weight_matrix() -> _NDArray:
    if filter.name == 'impulse':
      return np.ones(src_index.shape, dtype=dtype)
    if is_minification:
      x = (src_float_index[:, None] - src_index.astype(dtype)) * scaling
      return filter(x) * scaling
    # Either same size or magnification.
    x = src_float_index[:, None] - src_index.astype(dtype)
    return filter(x)

  weight = get_weight_matrix().astype(dtype, copy=False)

  if filter.name != 'narrowbox' and (is_minification or not filter.partition_of_unity):
    weight = weight / weight.sum(axis=-1)[..., None]

  src_index, weight = boundary.apply(src_index, weight, src_position, src_size, src_gridtype)
  shape = dst_size, src_size

  uses_cval = boundary.uses_cval or filter.name == 'narrowbox'
  cval_weight = 1.0 - weight.sum(axis=-1) if uses_cval else None

  # Create the sparse resize matrix.
  if arraylib == 'numpy':

    def numpy_create_sparse_matrix() -> _NDArray:
      data = weight.reshape(-1)
      row_ind = np.arange(dst_size).repeat(src_index.shape[1])
      col_ind = src_index.reshape(-1)
      nonzero = data != 0.0
      data, row_ind, col_ind = data[nonzero], row_ind[nonzero], col_ind[nonzero]
      # Note: csr_matrix automatically reorders and merges duplicate indices.
      return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)

    resize_matrix = numpy_create_sparse_matrix()

  elif arraylib == 'tensorflow':
    import tensorflow as tf

    def tensorflow_create_sparse_matrix() -> _TensorflowTensor:
      linearized = (src_index + np.indices(src_index.shape)[0] * src_size).reshape(-1)
      values = weight.reshape(-1)
      # Remove the zero weights.
      nonzero = values != 0.0
      linearized, values = linearized[nonzero], values[nonzero]
      # Sort and merge the duplicate indices.
      unique, unique_inverse = np.unique(linearized, return_inverse=True)
      data = np.ones(len(linearized), dtype=np.float32)
      row_ind = unique_inverse
      col_ind = np.arange(len(linearized))
      shape2 = len(unique), len(linearized)
      csr = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape2)
      merged_values = csr * values
      merged_indices = np.vstack((unique // src_size, unique % src_size))
      return tf.sparse.SparseTensor(merged_indices.T, merged_values, shape)

    resize_matrix = tensorflow_create_sparse_matrix()
    if cval_weight is not None:
      cval_weight = tf.convert_to_tensor(cval_weight)

  elif arraylib == 'torch':
    import torch

    def torch_create_sparse_matrix() -> torch.Tensor:
      indices = np.vstack((np.arange(dst_size).repeat(src_index.shape[1]),
                           src_index.reshape(-1))).T
      values = weight.reshape(-1)
      # Remove the zero weights, then coalesce the duplicate indices.
      nonzero = values != 0.0
      indices, values = indices[nonzero], values[nonzero]
      return torch.sparse_coo_tensor(indices.T, values, shape).coalesce()  # type: ignore[arg-type]

    resize_matrix = torch_create_sparse_matrix()
    if cval_weight is not None:
      cval_weight = torch.as_tensor(cval_weight)

  else:
    raise AssertionError(f'{arraylib} is unrecognized.')

  return resize_matrix, cval_weight


def _best_order_in_which_to_process_dimensions(
    array: _Array, dst_shape: Tuple[int, ...]) -> List[int]:
  """Return the best order in which to process dimensions when resizing `array` to `dst_shape`.

  For numpy: (1) a dimension with small scaling (especially minification) gets priority, and
             (2) timings show preference to resizing dimensions with larger strides first.

  For tensorflow: process dimension 1 first iff dimension 0 is upsampling.

  For torch: same as numpy.
  """
  # The optimal ordering might be related to the logic in np.einsum_path().  (Unfortunately,
  # np.einsum() does not support the sparse multiplications that we require here.)
  src_shape: Tuple[int, ...] = array.shape[:len(dst_shape)]
  arraylib = _arr_arraylib(array)
  strides: Sequence[int] = (array.strides if arraylib == 'numpy' else
                            list(reversed(range(len(dst_shape)))) if arraylib == 'tensorflow' else
                            array.stride() if arraylib == 'torch' else
                            1/0)
  largest_stride_dim = max(range(len(src_shape)), key=lambda dim: strides[dim])

  def priority(dim: int) -> float:
    scaling = dst_shape[dim] / src_shape[dim]
    if arraylib == 'numpy':
      return scaling * ((0.49 if scaling < 1.0 else 0.65) if dim == largest_stride_dim else 1.0)
    if arraylib == 'tensorflow':
      return 2.0 if scaling > 1.0 and dim == largest_stride_dim else 1.0
    if arraylib == 'torch':
      return scaling * ((0.49 if scaling < 1.0 else 0.65) if dim == largest_stride_dim else 1.0)
    raise AssertionError(f'{arraylib} is unrecognized.')

  dim_order = sorted(range(len(src_shape)), key=priority)
  return dim_order


def _apply_potential_digital_filter_1d(  # pylint: disable=too-many-statements
    array: _Array, gridtype: Gridtype, boundary: Boundary, cval: Any, filter: Filter,
    axis: int = 0, compute_backward: bool = False) -> _Array:
  """Apply inverse convolution to the specified dimension of the array.

  Find the array coefficients such that convolution with the (continuous) filter (given
  gridtype and boundary) interpolates the original array values.
  """
  if not filter.requires_digital_filter:
    return array
  arraylib = _arr_arraylib(array)

  if arraylib == 'tensorflow':
    import tensorflow as tf

    def forward(x: _NDArray) -> _NDArray:
      return _apply_potential_digital_filter_1d(x, gridtype, boundary, cval, filter, axis)

    def backward(grad_output: _NDArray) -> _NDArray:
      return _apply_potential_digital_filter_1d(grad_output, gridtype, boundary, cval, filter,
                                                axis, compute_backward=True)

    @tf.custom_gradient
    def inverse_convolution(x: Any) -> _TensorflowTensor:
      y = tf.numpy_function(forward, [x], x.dtype)

      def grad(dy: Any) -> _TensorflowTensor:
        return tf.numpy_function(backward, [dy], x.dtype)

      return y, grad

    return inverse_convolution(array)

  if arraylib == 'torch':
    import torch

    class InverseConvolution(torch.autograd.Function):
      """Differentiable wrapper for _apply_potential_digital_filter_1d."""

      @staticmethod
      def forward(unused_ctx: Any, x: Any) -> torch.Tensor:  # type: ignore[override]
        return torch.as_tensor(_apply_potential_digital_filter_1d(
            x.detach().numpy(), gridtype, boundary, cval, filter, axis))

      @staticmethod
      def backward(unused_ctx: Any, grad_output: Any) -> torch.Tensor:  # type: ignore[override]
        return torch.as_tensor(_apply_potential_digital_filter_1d(
            grad_output.detach().numpy(), gridtype, boundary, cval, filter,
            axis, compute_backward=True))

    return InverseConvolution.apply(array)

  assert np.issubdtype(array.dtype, np.inexact)
  cval = np.asarray(cval).astype(array.dtype, copy=False)

  # Use faster code if compatible dtype, gridtype, boundary, and filter:
  # ?? if (filter.name in ('cardinal3', 'cardinal5') and
  use_scipy_spline_filter1d = (
      isinstance(filter, CardinalBsplineFilter) and
      filter.degree >= 2 and
      not np.issubdtype(array.dtype, np.complexfloating) and (
          boundary.name == 'reflect' or (gridtype.name == 'dual' and boundary.name == 'wrap')))
  if use_scipy_spline_filter1d:
    assert isinstance(filter, CardinalBsplineFilter)
    # ?? order = int(filter.name[len('cardinal'):])
    mode = ({'dual': 'reflect', 'primal': 'mirror'}[gridtype.name]
            if boundary.name == 'reflect' else 'wrap')
    # compute_backward=True is same: matrix is symmetric and cval is unused.
    return _spline_filter1d(array, axis=axis, order=filter.degree, mode=mode, output=array.dtype)

  array_dim = np.moveaxis(array, axis, 0)
  l = original_l = math.ceil(filter.radius) - 1
  x = np.arange(-l, l + 1, dtype=array.real.dtype)
  values = filter(x)
  size = array_dim.shape[0]
  src_index = np.arange(size)[:, None] + np.arange(len(values)) - l
  weight = np.full((size, len(values)), values)
  src_position = np.broadcast_to(0.5, len(values))
  src_index, weight = boundary.apply(src_index, weight, src_position, size, gridtype)
  if gridtype.name == 'primal' and boundary.name == 'wrap':
    # Overwrite redundant last row to preserve unreferenced last sample and thereby make the
    # matrix non-singular.
    src_index[-1] = [size - 1] + [0] * (src_index.shape[1] - 1)
    weight[-1] = [1.0] + [0.0] * (weight.shape[1] - 1)
  bandwidth = abs(src_index - np.arange(size)[:, None]).max()
  is_banded = bandwidth <= l + 1  # Add one for quadratic boundary and l == 1.
  # Currently, matrix is always banded unless boundary.name == 'wrap'.

  data = weight.reshape(-1).astype(array.dtype, copy=False)
  row_ind = np.arange(size).repeat(src_index.shape[1])
  col_ind = src_index.reshape(-1)
  matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(size, size))
  if compute_backward:
    matrix = matrix.T

  if boundary.uses_cval and not compute_backward:
    cval_weight = 1.0 - np.asarray(matrix.sum(axis=-1))[:, 0]
    if array_dim.ndim == 2:  # Handle the case that we have array_flat.
      cval = np.tile(cval.reshape(-1), array_dim[0].size // cval.size)
    array_dim = array_dim - cval_weight.reshape(-1, *(1,) * array_dim[0].ndim) * cval

  if is_banded:
    matrix = matrix.todia()
    assert np.all(np.diff(matrix.offsets) == 1)  # Consecutive, often [-l, l].
    l, u = -matrix.offsets[0], matrix.offsets[-1]
    assert l <= original_l + 1 and u <= original_l + 1, (l, u, original_l)
    options = dict(check_finite=False, overwrite_ab=True, overwrite_b=False)
    if _is_symmetric(matrix):
      array_dim = scipy.linalg.solveh_banded(matrix.data[-1:l-1:-1], array_dim, **options)
    else:
      array_dim = scipy.linalg.solve_banded((l, u), matrix.data[::-1], array_dim, **options)

  else:
    lu = scipy.sparse.linalg.splu(matrix.tocsc(), permc_spec='NATURAL')
    assert all(s <= size * len(values) for s in (lu.L.nnz, lu.U.nnz))  # Sparse.
    array_dim = lu.solve(array_dim.reshape(array_dim.shape[0], -1)).reshape(array_dim.shape)

  return np.moveaxis(array_dim, 0, axis)


def resize(                     # pylint: disable=too-many-branches disable=too-many-statements
    array: _Array,
    shape: Iterable[int],
    *,
    gridtype: Union[None, str, Gridtype] = None,
    src_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]] = None,
    dst_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]] = None,
    boundary: Union[str, Boundary, Iterable[Union[str, Boundary]]] = 'auto',
    cval: Any = 0,
    filter: Union[str, Filter, Iterable[Union[str, Filter]]] = 'lanczos3',
    prefilter: Union[None, str, Filter, Iterable[Union[str, Filter]]] = None,
    gamma: Union[None, str, Gamma] = None,
    src_gamma: Union[None, str, Gamma] = None,
    dst_gamma: Union[None, str, Gamma] = None,
    scale: Union[float, Iterable[float]] = 1.0,
    translate: Union[float, Iterable[float]] = 0.0,
    precision: Any = None,
    dtype: Any = None,
    dim_order: Optional[Iterable[int]] = None,
    internal_torch_contiguous: bool = True,
) -> _Array:
  """Resample `array` (a grid of sample values) onto a grid with resolution `shape`.

  The source `array` may be an array-like, `np.ndarray`, `tf.Tensor`, or `torch.Tensor`.  The
  array is interpreted as a grid with `len(shape)` domain coordinate dimensions, where each
  grid sample value has shape `array.shape[len(shape):]`.

  For example:

  - A grayscale image has `array.shape = height, width` and resizing it with `len(shape) == 2`
    produces a new image of scalar values.
  - An RGB image has `array.shape = height, width, 3` and resizing it with `len(shape) == 2`
    produces a new image of RGB values.
  - An 3D grid of 3x3 Jacobians has `array.shape = Z, Y, X, 3, 3` and resizing it with
    `len(shape) == 3` produces a new 3D grid of Jacobians.

  This function also allows scaling and translation from the source domain to the output domain
  through the parameters `scale` and `translate`.  For more general transforms, see `resample`.

  Args:
    array: Grid of source sample values.  It must be an array-like object from a library in
      `ARRAYLIBS`.  The array must have numeric type.  Its first `len(shape)` dimensions are the
      domain coordinate dimensions.  Each grid dimension must be at least 1 for a `dual` grid or
      at least 2 for a `primal` grid.
    shape: The number of grid samples in each coordinate dimension of the output array.  The source
      `array` must have at least as many dimensions as `len(shape)`.
    gridtype: Placement of samples on all dimensions of both the source and output domain grids,
      specified as either a name in `GRIDTYPES` or a `Gridtype` instance.  The default is 'dual'.
    src_gridtype: Placement of the samples in the source domain grid for each dimension.
      Parameters `gridtype` and `src_gridtype` cannot both be set.
    dst_gridtype: Placement of the samples in the output domain grid for each dimension.
      Parameters `gridtype` and `dst_gridtype` cannot both be set.
    boundary: The reconstruction boundary rule for each dimension in `shape`, specified as either
      a name in `BOUNDARIES` or a `Boundary` instance.  The special value 'auto' uses 'reflect'
      for upsampling and 'clamp' for downsampling.
    cval: Constant value used beyond the samples by some boundary rules.  It must be broadcastable
      onto `array.shape[len(shape):]`.
    filter: The reconstruction kernel for each dimension in `shape`, specified as either a filter
      name in `FILTERS` or a `Filter` instance.  It is used during upsampling (i.e., magnification).
    prefilter: The prefilter kernel for each dimension in `shape`, specified as either a filter
      name in `FILTERS` or a `Filter` instance.  It is used during downsampling
      (i.e., minification).  If `None`, it inherits the value of `filter`.
    gamma: Component transfer functions (e.g., gamma correction) applied when reading samples from
      `array` and when creating output grid samples.  It is specified as either a name in `GAMMAS`
      or a `Gamma` instance.  If both `array.dtype` and `dtype` are `uint`, the default is `power2`.
      If both are non-`uint`, the default is `identity`.  Otherwise, `gamma` or
      `src_gamma`/`dst_gamma` must be set.   Gamma correction assumes that float values are in the
      range [0.0, 1.0].
    src_gamma: Component transfer function used to "decode" `array` samples.
      Parameters `gamma` and `src_gamma` cannot both be set.
    dst_gamma: Component transfer function used to "encode" the output samples.
      Parameters `gamma` and `dst_gamma` cannot both be set.
    scale: Scaling factor applied to each dimension of the source domain when it is mapped onto
      the destination domain.
    translate: Offset applied to each dimension of the scaled source domain when it is mapped onto
      the destination domain.
    precision: Inexact precision of intermediate computations.  If `None`, it is determined based
      on `array.dtype` and `dtype`.
    dtype: Desired data type of the output array.  If `None`, it is taken to be `array.dtype`.
      If it is a uint type, the intermediate float values are rescaled from the [0.0, 1.0] range
      to the uint range.
    dim_order: Override the automatically selected order in which the grid dimensions are resized.
      Must contain a permutation of `range(len(shape))`.

  Returns:
    An array of the same class (`np.ndarray`, `tf.Tensor`, or `torch.Tensor`) as the source `array`,
    with shape `shape + array.shape[len(shape):]` and data type `dtype`.

  >>> result = resize([1.0, 4.0, 5.0], shape=(4,))
  >>> assert np.allclose(result, [0.74240461, 2.88088827, 4.68647155, 5.02641199])
  """
  if isinstance(array, (tuple, list)):
    array = np.asarray(array)
  arraylib = _arr_arraylib(array)
  array_dtype = _arr_dtype(array)
  if not np.issubdtype(array_dtype, np.number):
    raise ValueError(f'Type {array.dtype} is not numeric.')
  shape = tuple(shape)
  array_ndim = len(array.shape)
  if not 0 < len(shape) <= array_ndim:
    raise ValueError(f'Shape {array.shape} cannot be resized to {shape}.')
  src_shape = array.shape[:len(shape)]
  src_gridtype2, dst_gridtype2 = _get_gridtypes(
      gridtype, src_gridtype, dst_gridtype, len(shape), len(shape))
  boundary2 = np.broadcast_to(np.array(boundary), len(shape))
  cval = np.broadcast_to(cval, array.shape[len(shape):])
  prefilter = filter if prefilter is None else prefilter
  filter2 = [_get_filter(f) for f in np.broadcast_to(np.array(filter), len(shape))]
  prefilter2 = [_get_filter(f) for f in np.broadcast_to(np.array(prefilter), len(shape))]
  dtype = array_dtype if dtype is None else np.dtype(dtype)
  src_gamma2, dst_gamma2 = _get_src_dst_gamma(gamma, src_gamma, dst_gamma, array_dtype, dtype)
  scale2 = np.broadcast_to(np.array(scale), len(shape))
  translate2 = np.broadcast_to(np.array(translate), len(shape))
  del src_gridtype, dst_gridtype, boundary, filter, prefilter
  del src_gamma, dst_gamma, scale, translate
  precision = _get_precision(precision, [array_dtype, dtype], [])
  weight_precision = _real_precision(precision)
  if dim_order is None:
    dim_order = _best_order_in_which_to_process_dimensions(array, shape)
  else:
    dim_order = tuple(dim_order)
    if sorted(dim_order) != list(range(len(shape))):
      raise ValueError(f'{dim_order} not a permutation of {list(range(len(shape)))}.')

  array = src_gamma2.decode(array, precision)

  can_use_fast_box_downsampling = (
      'numba' in globals() and arraylib == 'numpy' and len(shape) == 2 and array_ndim in (2, 3) and
      all(src > dst for src, dst in zip(src_shape, shape)) and
      all(src % dst == 0 for src, dst in zip(src_shape, shape)) and
      all(gridtype.name == 'dual' for gridtype in src_gridtype2) and
      all(gridtype.name == 'dual' for gridtype in dst_gridtype2) and
      all(f.name in ('box', 'trapezoid') for f in prefilter2) and
      np.all(scale2 == 1.0) and np.all(translate2 == 0.0))
  if can_use_fast_box_downsampling:
    array = _downsample_in_2d_using_box_filter(array, typing.cast(Tuple[int, int], shape))
    array = dst_gamma2.encode(array, dtype=dtype)
    return array

  for dim in dim_order:
    skip_resize_on_this_dim = (shape[dim] == array.shape[dim] and scale2[dim] == 1.0 and
                               translate2[dim] == 0.0 and filter2[dim].interpolating)
    if skip_resize_on_this_dim:
      continue

    is_minification = (dst_gridtype2[dim].size_in_samples(shape[dim]) /
                       src_gridtype2[dim].size_in_samples(array.shape[dim])) * scale2[dim] < 1.0
    boundary_dim = boundary2[dim]
    if boundary_dim == 'auto':
      boundary_dim = 'clamp' if is_minification else 'reflect'
    boundary_dim = _get_boundary(boundary_dim)
    resize_matrix, cval_weight = _create_resize_matrix(
        array.shape[dim],
        shape[dim],
        src_gridtype=src_gridtype2[dim],
        dst_gridtype=dst_gridtype2[dim],
        boundary=boundary_dim,
        filter=filter2[dim],
        prefilter=prefilter2[dim],
        scale=scale2[dim],
        translate=translate2[dim],
        dtype=weight_precision,
        arraylib=arraylib)

    def potential_pre_digital_filter(array_flat: Any) -> _Array:
      if not is_minification:
        array_flat = _apply_potential_digital_filter_1d(
            array_flat, src_gridtype2[dim], boundary_dim, cval, filter2[dim])
      return array_flat

    def potential_post_digital_filter(array_flat: Any) -> _Array:
      if is_minification:  # use prefilter2[dim]?
        array_flat = _apply_potential_digital_filter_1d(
            array_flat, dst_gridtype2[dim], boundary_dim, cval, filter2[dim])
      return array_flat

    if arraylib == 'numpy':

      def resize_numpy_dim() -> _NDArray:
        # Because np.ndarray supports strides, np.moveaxis() and np.permute() are constant-time.
        # However, ndarray.reshape() often creates a copy of the array if the data is
        # non-contiguous, e.g. dim=1 in an RGB image.
        array_dim = np.moveaxis(array, dim, 0)
        array_flat = array_dim.reshape(array_dim.shape[0], -1)
        array_flat = potential_pre_digital_filter(array_flat)

        # The dot() calls scipy.sparse._sparsetools.csr_matvecs() in
        # https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
        # which iteratively calls the LEVEL 1 BLAS function axpy().
        array_flat = resize_matrix.dot(array_flat)
        if cval_weight is not None:
          cval_flat = np.broadcast_to(cval, array_dim.shape[1:]).reshape(-1)
          array_flat += cval_weight[:, None] * cval_flat

        array_flat = potential_post_digital_filter(array_flat)
        array_dim = array_flat.reshape(array_flat.shape[0], *array_dim.shape[1:])
        return np.moveaxis(array_dim, 0, dim)

      array = resize_numpy_dim()

    elif arraylib == 'tensorflow':

      def resize_tensorflow_dim() -> _TensorflowTensor:
        import tensorflow as tf
        # Because tf.Tensor does not support strides, tf.transpose() returns a
        # new permuted tensor.  However, tf.reshape() is always efficient.
        perm = [dim] + list(range(dim)) + list(range(dim + 1, array_ndim))
        array_dim = tf.transpose(array, perm)
        array_flat = tf.reshape(array_dim, [array_dim.shape[0], -1])
        array_flat = potential_pre_digital_filter(array_flat)

        array_flat = tf.sparse.sparse_dense_matmul(resize_matrix, array_flat)
        if cval_weight is not None:
          cval_flat = np.broadcast_to(cval, array_dim.shape[1:]).reshape(-1)
          array_flat += cval_weight[:, None] * cval_flat

        array_flat = potential_post_digital_filter(array_flat)
        array_dim = tf.reshape(array_flat, [array_flat.shape[0], *array_dim.shape[1:]])
        perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, array_ndim))
        return tf.transpose(array_dim, perm)

      if np.issubdtype(array_dtype, np.complexfloating):
        resize_matrix = resize_matrix.with_values(_arr_astype(resize_matrix.values, array_dtype))
        if cval_weight is not None:
          cval_weight = _arr_astype(cval_weight, array_dtype)
      array = resize_tensorflow_dim()

    elif arraylib == 'torch':

      def resize_torch_dim() -> _TorchTensor:
        import torch
        # No sparse complex: https://github.com/pytorch/pytorch/issues/50690.
        assert not np.issubdtype(array_dtype, np.complexfloating)
        array_dim = array.moveaxis(dim, 0)
        array_flat: torch.Tensor = array_dim.reshape(array_dim.shape[0], -1)
        if internal_torch_contiguous:  # Greatly improves timings in some cases.
          array_flat = array_flat.contiguous()
        array_flat = potential_pre_digital_filter(array_flat)

        array_flat = torch.mm(resize_matrix, array_flat)
        if cval_weight is not None:
          cval_flat = np.broadcast_to(cval, array_dim.shape[1:]).reshape(-1)
          array_flat += cval_weight[:, None] * cval_flat

        array_flat = potential_post_digital_filter(array_flat)
        array_dim = array_flat.reshape(array_flat.shape[0], *array_dim.shape[1:])
        return array_dim.moveaxis(0, dim)

      array = resize_torch_dim()

    else:
      AssertionError(f'{arraylib} is unrecognized.')

  array = dst_gamma2.encode(array, dtype=dtype)
  return array


_original_resize = resize


def resize_in_numpy(array: _NDArray, *args: Any, **kwargs: Any) -> _NDArray:
  """Just like `resize()` but asserts that the source is a numpy array."""
  _check_eq(_arr_arraylib(array), 'numpy')
  return _original_resize(array, *args, **kwargs)


def resize_in_tensorflow(array: _NDArray, *args: Any, **kwargs: Any) -> _NDArray:
  """Evaluate the `resize()` operation using Tensorflow's Tensor representation and operations.

  Args:
    array: Grid of source samples, represented as a numpy array.
    *args: Parameters for `resize()`.
    **kwargs: Parameters for `resize()`.

  Returns:
    A numpy array.
  """
  _check_eq(_arr_arraylib(array), 'numpy')
  array = _make_array(array, arraylib='tensorflow')
  return _original_resize(array, *args, **kwargs).numpy()


def resize_in_torch(array: _NDArray, *args: Any, **kwargs: Any) -> _NDArray:
  """Evaluate the `resize()` operation using Torch's Tensor representation and operations.

  Args:
    array: Grid of source samples, represented as a numpy array.
    *args: Parameters for `resize()`.
    **kwargs: Parameters for `resize()`.

  Returns:
    A numpy array.
  """
  _check_eq(_arr_arraylib(array), 'numpy')
  array = _make_array(array, arraylib='torch')
  return _original_resize(array, *args, **kwargs).numpy()


_MAX_BLOCK_SIZE_RECURSING = -999  # Special value to indicate re-invocation on partitioned blocks.


def resample(                   # pylint: disable=too-many-branches disable=too-many-statements
    array: _Array,
    coords: Any,
    *,
    gridtype: Union[str, Gridtype, Iterable[Union[str, Gridtype]]] = 'dual',
    boundary: Union[str, Boundary, Iterable[Union[str, Boundary]]] = 'auto',
    cval: Any = 0,
    filter: Union[str, Filter, Iterable[Union[str, Filter]]] = 'lanczos3',
    prefilter: Union[None, str, Filter, Iterable[Union[str, Filter]]] = None,
    gamma: Union[None, str, Gamma] = None,
    src_gamma: Union[None, str, Gamma] = None,
    dst_gamma: Union[None, str, Gamma] = None,
    jacobian: Any = None,
    precision: Any = None,
    dtype: Any = None,
    max_block_size: int = 40_000,
    debug: bool = False,
) -> _NDArray:
  """Interpolate `array` (a grid of samples) at specified unit-domain coordinates `coords`.

  The last dimension of `coords` contains unit-domain coordinates at which to interpolate the
  domain grid samples in `array`.

  The number of coordinates (`coords.shape[-1]`) determines how to interpret `array`: its first
  `coords.shape[-1]` dimensions define the grid, and the remaining dimensions describe each grid
  sample (e.g., scalar, vector, tensor).

  Concretely, the grid has shape `array.shape[:coords.shape[-1]]` and each grid sample has shape
  `array.shape[coords.shape[-1]:]`.

  Examples include:

  - Resample a grayscale image with `array.shape = height, width` onto a new grayscale image with
    `new.shape = height2, width2` by using `coords.shape = height2, width2, 2`.

  - Resample an RGB image with `array.shape = height, width, 3` onto a new RGB image with
    `new.shape = height2, width2, 3` by using `coords.shape = height2, width2, 2`.

  - Sample an RGB image at `num` 2D points along a line segment by using `coords.shape = num, 2`.

  - Sample an RGB image at a single 2D point by using `coords.shape = (2,)`.

  - Sample a 3D grid of 3x3 Jacobians with `array.shape = nz, ny, nx, 3, 3` along a 2D plane by
    using `coords.shape = height, width, 3`.

  - Map a grayscale image through a color map by using `array.shape = 256, 3` and
    `coords.shape = height, width`.

  Args:
    array: Grid of source sample values.  It must be an array-like object from a library in
      `ARRAYLIBS`.  The array must have numeric type.  The coordinate dimensions appear first, and
      each grid sample may have an arbitrary shape.  Each grid dimension must be at least 1 for
      a `dual` grid or at least 2 for a `primal` grid.
    coords: Grid of points at which to resample `array`.  The point coordinates are in the last
      dimension of `coords`.  The domain associated with the source grid is a unit hypercube,
      i.e. with a range [0, 1] on each coordinate dimension.  The output grid has shape
      `coords.shape[:-1]` and each of its grid samples has shape `array.shape[coords.shape[-1]:]`.
    gridtype: Placement of the samples in the source domain grid for each dimension, specified as
      either a name in `GRIDTYPES` or a `Gridtype` instance.
    boundary: The reconstruction boundary rule for each dimension in `coords.shape[-1]`, specified
      as either a name in `BOUNDARIES` or a `Boundary` instance.  The special value 'auto' uses
      'reflect' for upsampling and 'clamp' for downsampling.
    cval: Constant value used beyond the samples by some boundary rules.  It must be broadcastable
      onto the shape `array.shape[coords.shape[-1]:]`.
    filter: The reconstruction kernel for each dimension in `coords.shape[-1]`, specified as either
      a filter name in `FILTERS` or a `Filter` instance.
    prefilter: The prefilter kernel for each dimension in `coords.shape[:-1]`, specified as either
      a filter name in `FILTERS` or a `Filter` instance.  It is used during downsampling
      (i.e., minification).  If `None`, it inherits the value of `filter`.
    gamma: Component transfer functions (e.g., gamma correction) applied when reading samples
      from `array` and when creating output grid samples.  It is specified as either a name in
      `GAMMAS` or a `Gamma` instance.  If both `array.dtype` and `dtype` are `uint`, the default
      is `power2`.  If both are non-`uint`, the default is `identity`.  Otherwise, `gamma` or
      `src_gamma`/`dst_gamma` must be set.   Gamma correction assumes that float values are in the
      range [0.0, 1.0].
    src_gamma: Component transfer function used to "decode" `array` samples.
      Parameters `gamma` and `src_gamma` cannot both be set.
    dst_gamma: Component transfer function used to "encode" the output samples.
      Parameters `gamma` and `dst_gamma` cannot both be set.
    jacobian: Optional array, which must be broadcastable onto the shape
      `coords.shape[:-1] + (coords.shape[-1], coords.shape[-1])`, storing for each point in the
      output grid the Jacobian matrix of the map from the unit output domain to the unit source
      domain.  If omitted, it is estimated by computing finite differences on `coords`.
    precision: Inexact precision of intermediate computations.  If `None`, it is determined based
      on `array.dtype`, `coords.dtype`, and `dtype`.
    dtype: Desired data type of the output array.  If `None`, it is taken to be `array.dtype`.
      If it is a uint type, the intermediate float values are rescaled from the [0.0, 1.0] range
      to the uint range.
    max_block_size: If nonzero, maximum number of grid points in `coords` before the resampling
      evaluation gets partitioned into smaller blocks for reduced memory usage and better caching.
    debug: Show internal information.

  Returns:
    A new sample grid of shape `coords.shape[:-1]`, represented as an array of shape
    `coords.shape[:-1] + array.shape[coords.shape[-1]:]`, of the same array library type as
    the source array.

  For reference, the identity resampling for a scalar-valued grid with the default grid-type
  'dual' is:

  >>> array = np.random.default_rng(0).random((5, 7, 3))
  >>> coords = (np.moveaxis(np.indices(array.shape), 0, -1) + 0.5) / array.shape
  >>> new_array = resample(array, coords)
  >>> assert np.allclose(new_array, array)

  It is more efficient to use the function `resize` for the special case where the `coords` are
  obtained as simple scaling and translation of a new regular grid over the source domain:

  >>> scale, translate, new_shape = (1.1, 1.2), (0.1, -0.2), (6, 8)
  >>> coords = (np.moveaxis(np.indices(new_shape), 0, -1) + 0.5) / new_shape
  >>> coords = (coords - translate) / scale
  >>> resampled = resample(array, coords)
  >>> resized = resize(array, new_shape, scale=scale, translate=translate)
  >>> assert np.allclose(resampled, resized)
  """
  if isinstance(array, (tuple, list)):
    array = np.asarray(array)
  arraylib = _arr_arraylib(array)
  if len(array.shape) == 0:
    array = array[None]
  coords = np.atleast_1d(coords)
  if not np.issubdtype(_arr_dtype(array), np.number):
    raise ValueError(f'Type {array.dtype} is not numeric.')
  if not np.issubdtype(coords.dtype, np.floating):
    raise ValueError(f'Type {coords.dtype} is not floating.')
  array_ndim = len(array.shape)
  if coords.ndim == 1 and coords.shape[0] > 1 and array_ndim == 1:
    coords = coords[:, None]
  grid_ndim = coords.shape[-1]
  grid_shape = array.shape[:grid_ndim]
  sample_shape = array.shape[grid_ndim:]
  resampled_ndim = coords.ndim - 1
  resampled_shape = coords.shape[:-1]
  if grid_ndim > array_ndim:
    raise ValueError(f'There are more coordinate dimensions ({grid_ndim}) in'
                     f' coords {coords} than in array.shape {array.shape}.')
  gridtype2 = [_get_gridtype(g) for g in np.broadcast_to(np.array(gridtype), grid_ndim)]
  boundary2 = np.broadcast_to(np.array(boundary), grid_ndim).tolist()
  cval = np.broadcast_to(cval, sample_shape)
  prefilter = filter if prefilter is None else prefilter
  filter2 = [_get_filter(f) for f in np.broadcast_to(np.array(filter), grid_ndim)]
  prefilter2 = [_get_filter(f) for f in np.broadcast_to(np.array(prefilter), resampled_ndim)]
  dtype = _arr_dtype(array) if dtype is None else np.dtype(dtype)
  src_gamma2, dst_gamma2 = _get_src_dst_gamma(gamma, src_gamma, dst_gamma, _arr_dtype(array), dtype)
  del gridtype, boundary, filter, prefilter, src_gamma, dst_gamma
  if jacobian is not None:
    jacobian = np.broadcast_to(jacobian, resampled_shape + (coords.shape[-1],) * 2)
  precision = _get_precision(precision, [_arr_dtype(array), dtype], [coords.dtype])
  weight_precision = _real_precision(precision)
  coords = coords.astype(weight_precision, copy=False)
  is_minification = False  # Current limitation; no prefiltering!
  assert max_block_size >= 0 or max_block_size == _MAX_BLOCK_SIZE_RECURSING
  for dim in range(grid_ndim):
    if boundary2[dim] == 'auto':
      boundary2[dim] = 'clamp' if is_minification else 'reflect'
    boundary2[dim] = _get_boundary(boundary2[dim])

  if max_block_size != _MAX_BLOCK_SIZE_RECURSING:
    array = src_gamma2.decode(array, precision)
    for dim in range(grid_ndim):
      assert not is_minification
      array = _apply_potential_digital_filter_1d(
          array, gridtype2[dim], boundary2[dim], cval, filter2[dim], axis=dim)

  if np.prod(resampled_shape) > max_block_size > 0:
    block_shape = _block_shape_with_min_size(resampled_shape, max_block_size)
    if debug:
      print(f'(resample: splitting coords into blocks {block_shape}).')
    coord_blocks = _split_array_into_blocks(coords, block_shape)

    def process_block(coord_block: _NDArray) -> _NDArray:
      return resample(
          array, coord_block, gridtype=gridtype2, boundary=boundary2, cval=cval,
          filter=filter2, prefilter=prefilter2, src_gamma='identity', dst_gamma=dst_gamma2,
          jacobian=jacobian, precision=precision, dtype=dtype,
          max_block_size=_MAX_BLOCK_SIZE_RECURSING)

    result_blocks = _map_function_over_blocks(coord_blocks, process_block)
    array = _merge_array_from_blocks(result_blocks)
    return array

  # A concrete example of upsampling:
  #   array = np.ones((5, 7, 3))  # source RGB image has height=5 width=7
  #   coords = np.random.default_rng(0).random((8, 9, 2))  # output RGB image has height=8 width=9
  #   resample(array, coords, filter=('cubic', 'lanczos3'))
  #   grid_shape = 5, 7  grid_ndim = 2
  #   resampled_shape = 8, 9  resampled_ndim = 2
  #   sample_shape = (3,)
  #   src_float_index.shape = 8, 9
  #   src_first_index.shape = 8, 9
  #   sample_index.shape = (4,) for dim == 0, then (6,) for dim == 1
  #   weight = [shape(8, 9, 4), shape(8, 9, 6)]
  #   src_index = [shape(8, 9, 4), shape(8, 9, 6)]

  # Both:[shape(8, 9, 4), shape(8, 9, 6)]
  weight: List[_NDArray] = [np.array([]) for _ in range(grid_ndim)]
  src_index: List[_NDArray] = [np.array([]) for _ in range(grid_ndim)]
  uses_cval = False
  all_num_samples = []  # will be [4, 6]

  for dim in range(grid_ndim):
    src_size = grid_shape[dim]  # scalar
    coords_dim = coords[..., dim]  # (8, 9)
    radius = filter2[dim].radius  # scalar
    num_samples = int(np.ceil(radius * 2))  # scalar
    all_num_samples.append(num_samples)

    boundary_dim = boundary2[dim]
    coords_dim = boundary_dim.preprocess_coordinates(coords_dim)

    # Sample positions mapped back to source unit domain [0, 1].
    src_float_index = gridtype2[dim].index_from_point(coords_dim, src_size)  # (8, 9)
    src_first_index = (
        np.floor(src_float_index + (0.5 if num_samples % 2 == 1 else 0.0)).astype(np.int32)
        - (num_samples - 1) // 2)  # (8, 9)

    sample_index = np.arange(num_samples, dtype=np.int32)  # (4,) then (6,)
    src_index[dim] = src_first_index[..., None] + sample_index  # (8, 9, 4) then (8, 9, 6)
    if filter2[dim].name == 'trapezoid':
      # (It might require changing the filter radius at every sample.)
      raise ValueError('resample() cannot use adaptive `trapezoid` filter.')
    if filter2[dim].name == 'impulse':
      weight[dim] = np.ones_like(src_index[dim], dtype=weight_precision)
    else:
      x = src_float_index[..., None] - src_index[dim].astype(weight_precision)
      weight[dim] = filter2[dim](x).astype(weight_precision, copy=False)
      if (filter2[dim].name != 'narrowbox' and
          (is_minification or not filter2[dim].partition_of_unity)):
        weight[dim] = weight[dim] / weight[dim].sum(axis=-1)[..., None]

    src_index[dim], weight[dim] = boundary_dim.apply(
        src_index[dim], weight[dim], coords_dim, src_size, gridtype2[dim])
    if boundary_dim.uses_cval or filter2[dim].name == 'narrowbox':
      uses_cval = True

  # match arraylib:
  if arraylib == 'tensorflow':
    import tensorflow as tf
    # Recall that src_index = [shape(8, 9, 4), shape(8, 9, 6)].
    gather_shape = resampled_shape + tuple(a.shape[-1] for a in src_index)  # (8, 9, 4, 6)
    src_index_expanded = []
    for dim in range(grid_ndim):
      src_index_dim = np.moveaxis(
          src_index[dim].reshape(src_index[dim].shape + (1,) * (grid_ndim - 1)),
          resampled_ndim, resampled_ndim + dim)
      src_index_expanded.append(np.broadcast_to(src_index_dim, gather_shape))
    # Unfortunately, broadcast expansion seems unavoidable for tf.gather_nd().
    indices = np.moveaxis(np.array(src_index_expanded), 0, -1)  # (8, 9, 4, 6, 2)
    samples = tf.gather_nd(array, indices)  # (8, 9, 4, 6, 3)

  else:  # 'numpy', 'torch'.
    # Recall that src_index = [shape(8, 9, 4), shape(8, 9, 6)].
    src_index_expanded = []  # [(8, 9, 4, 1), (8, 9, 1, 6)]
    for dim in range(grid_ndim):
      src_index_expanded.append(np.moveaxis(
          src_index[dim].reshape(src_index[dim].shape + (1,) * (grid_ndim - 1)),
          resampled_ndim, resampled_ndim + dim))
    # Unfortunately, gathering 'samples' is the memory bottleneck.  It would be ideal if we could
    # defer the array reads until inside the np.einsum() call.
    # For large outputs, splitting the evaluation over output tiles helps a bit.
    samples = array[tuple(src_index_expanded)]  # (8, 9, 4, 6, 3)

  samples_ndim = len(samples.shape)

  # Compute an Einstein summation over the samples and each of the
  # per-dimension weights.

  def label(dims: Iterable[int]) -> str:
    return ''.join(chr(ord('a') + i) for i in dims)

  operands = [samples]  # (8, 9, 4, 6, 3)
  labels = [label(range(samples_ndim))]  # ['abcde']
  for dim in range(grid_ndim):
    operands.append(weight[dim])  # (8, 9, 4), then (8, 9, 6)
    labels.append(label(list(range(resampled_ndim)) + [resampled_ndim + dim]))  # 'abc' then 'abd'
  output_label = label(list(range(resampled_ndim)) +
                       list(range(resampled_ndim + grid_ndim, samples_ndim)))  # 'abe'
  subscripts = ','.join(labels) + '->' + output_label  # 'abcde,abc,abd->abe'

  if arraylib == 'numpy':
    array = np.einsum(subscripts, *operands, optimize=True)  # (8, 9, 3)

  elif arraylib == 'tensorflow':
    import tensorflow as tf
    array = tf.einsum(subscripts, *operands, optimize='greedy')  # (8, 9, 3)

  elif arraylib == 'torch':
    import torch
    operands = [torch.as_tensor(operand) for operand in operands]
    array = torch.einsum(subscripts, *operands)  # (8, 9, 3)

  else:
    raise AssertionError(f'{arraylib} is unrecognized.')

  if uses_cval:
    cval_weight = 1.0 - np.multiply.reduce(
        [weight[dim].sum(axis=-1) for dim in range(resampled_ndim)])  # (8, 9)
    cval_weight_reshaped = cval_weight.reshape(cval_weight.shape + (1,) * len(sample_shape))
    array += _make_array((cval_weight_reshaped * cval).astype(precision, copy=False), arraylib)

  if max_block_size != _MAX_BLOCK_SIZE_RECURSING:
    array = dst_gamma2.encode(array, dtype=dtype)

  return array


def resample_affine(
    array: _Array,
    shape: Iterable[int],
    matrix: Any,
    *,
    gridtype: Union[None, str, Gridtype] = None,
    src_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]] = None,
    dst_gridtype: Union[None, str, Gridtype, Iterable[Union[str, Gridtype]]] = None,
    filter: Union[str, Filter, Iterable[Union[str, Filter]]] = 'lanczos3',
    prefilter: Union[None, str, Filter, Iterable[Union[str, Filter]]] = None,
    precision: Any = None,
    dtype: Any = None,
    **kwargs: Any,
) -> _NDArray:
  """Resample a source array using an affinely transformed grid of given shape.

  The `matrix` transformation can be linear:
    source_point = matrix @ destination_point.
  or it can be affine where the last matrix column is an offset vector:
    source_point = matrix @ (destination_point, 1.0)

  Args:
    array: Regular grid of source sample values.  The array must have numeric type.  The number of
      grid dimensions is determined from `matrix.shape[0]`; the remaining dimensions are for each
      sample value and are all linearly interpolated.
    shape: Dimensions of the desired destination grid.  The number of destination grid dimensions
      may be different from that of the source grid.
    matrix: 2D array for a linear or affine transform from unit-domain destination points
      (in a space with `len(shape)` dimensions) into unit-domain source points (in a space with
      `matrix.shape[0]` dimensions).  If the matrix has `len(shape) + 1` columns, the last column
      is the affine offset (i.e., translation).
    gridtype: Placement of samples on both the source and output domain grids, specified as either
      a name in `GRIDTYPES` or a `Gridtype` instance.  The default is 'dual'.
    src_gridtype: Placement of samples in the source domain grid for each dimension.
      Parameters `gridtype` and `src_gridtype` cannot both be set.
    dst_gridtype: Placement of samples in the output domain grid for each dimension.
      Parameters `gridtype` and `dst_gridtype` cannot both be set.
    filter: The reconstruction kernel for each dimension in `matrix.shape[0]`, specified as either
      a filter name in `FILTERS` or a `Filter` instance.
    prefilter: The prefilter kernel for each dimension in `len(shape)`, specified as either
      a filter name in `FILTERS` or a `Filter` instance.  It is used during downsampling
      (i.e., minification).  If `None`, it inherits the value of `filter`.
    precision: Inexact precision of intermediate computations.  If `None`, it is determined based
      on `array.dtype` and `dtype`.
    dtype: Desired data type of the output array.  If `None`, it is taken to be `array.dtype`.
      If it is a uint type, the intermediate float values are rescaled from the [0.0, 1.0] range
      to the uint range.
    **kwargs: Additional parameters for `resample` function.

  Returns:
    An array of the same class (`np.ndarray`, `tf.Tensor`, or `torch.Tensor`) as the source `array`,
    representing a grid with specified `shape`, where each grid value is resampled from `array`.
    Thus the shape of the returned array is `shape + array.shape[matrix.shape[0]:]`.
  """
  if isinstance(array, (tuple, list)):
    array = np.asarray(array)
  shape = tuple(shape)
  matrix = np.asarray(matrix)
  dst_ndim = len(shape)
  if matrix.ndim != 2:
    raise ValueError(f'Array {matrix} is not 2D matrix.')
  src_ndim = matrix.shape[0]
  # grid_shape = array.shape[:src_ndim]
  is_affine = matrix.shape[1] == dst_ndim + 1
  if src_ndim > len(array.shape):
    raise ValueError(f'Matrix {matrix} has more rows ({matrix.shape[0]}) than'
                     f' ndim in array.shape={array.shape}.')
  if matrix.shape[1] != dst_ndim and not is_affine:
    raise ValueError(f'Matrix has shape {matrix.shape}, but we expect either'
                     f' {dst_ndim} or {dst_ndim + 1} columns.')
  src_gridtype2, dst_gridtype2 = _get_gridtypes(
    gridtype, src_gridtype, dst_gridtype, src_ndim, dst_ndim)
  prefilter = filter if prefilter is None else prefilter
  filter2 = [_get_filter(f) for f in np.broadcast_to(np.array(filter), src_ndim)]
  prefilter2 = [_get_filter(f) for f in np.broadcast_to(np.array(prefilter), dst_ndim)]
  del src_gridtype, dst_gridtype, filter, prefilter
  dtype = _arr_dtype(array) if dtype is None else np.dtype(dtype)
  precision = _get_precision(precision, [_arr_dtype(array), dtype], [])
  weight_precision = _real_precision(precision)

  dst_position_list = []  # per dimension
  for dim in range(dst_ndim):
    dst_size = shape[dim]
    dst_index = np.arange(dst_size, dtype=weight_precision)
    dst_position_list.append(dst_gridtype2[dim].point_from_index(dst_index, dst_size))
  dst_position = np.meshgrid(*dst_position_list, indexing='ij')

  linear_matrix = matrix[:, :-1] if is_affine else matrix
  src_position = np.tensordot(linear_matrix, dst_position, 1)
  coords = np.moveaxis(src_position, 0, -1)
  if is_affine:
    coords += matrix[:, -1]

  # TODO: Based on grid_shape, shape, linear_matrix, and prefilter, determine a
  # convolution prefilter and apply it to bandlimit 'array', using boundary for padding.

  return resample(array, coords, gridtype=src_gridtype2, filter=filter2, prefilter=prefilter2,
                  precision=precision, dtype=dtype, **kwargs)


def resize_using_resample(array: _Array, shape: Iterable[int], *,
                          scale: Any = 1.0, translate: Any = 0.0, fallback: bool = False,
                          **kwargs: Any) -> _Array:
  """Use the more general `resample` operation for `resize`, as a debug tool."""
  if isinstance(array, (tuple, list)):
    array = np.asarray(array)
  shape = tuple(shape)
  scale = np.broadcast_to(scale, len(shape))
  translate = np.broadcast_to(translate, len(shape))
  # TODO: let resample() do prefiltering for proper downsampling.
  has_minification = np.any(np.array(shape) < array.shape[:len(shape)]) or np.any(scale < 1.0)
  if fallback and has_minification:
    return _original_resize(array, shape, scale=scale, translate=translate, **kwargs)
  offset = -translate / scale
  matrix = np.concatenate([np.diag(1.0 / scale), offset[:, None]], axis=1)
  return resample_affine(array, shape, matrix, **kwargs)


def rotation_about_center_in_2d(src_shape: Any,
                                angle: float,
                                dst_shape: Any = None,
                                scale: float = 1.0) -> _NDArray:
  """Return the 3x3 matrix mapping destination into a source unit domain.

  The returned matrix accounts for the possibly non-square domain shapes.

  Args:
    src_shape: Resolution (ny, nx) of the source domain grid.
    angle: Angle in radians (positive from x to y axis) applied when mapping the source domain
      onto the destination domain.
    dst_shape: Resolution (ny, nx) of the destination domain grid; it defaults to `src_shape`.
    scale: Scaling factor applied when mapping the source domain onto the destination domain.
  """

  def translation_matrix(vector: Any) -> _NDArray:
    matrix = np.eye(len(vector) + 1)
    matrix[:-1, -1] = vector
    return matrix

  def scaling_matrix(scale: Any) -> _NDArray:
    return np.diag(tuple(scale) + (1.0,))

  def rotation_matrix_2d(angle: float) -> _NDArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

  src_shape = np.asarray(src_shape)
  dst_shape = src_shape if dst_shape is None else np.asarray(dst_shape)
  _check_eq(src_shape.shape, (2,))
  _check_eq(dst_shape.shape, (2,))
  half = np.array([0.5, 0.5])
  matrix = (translation_matrix(half) @
            scaling_matrix(min(src_shape) / src_shape) @
            rotation_matrix_2d(angle) @
            scaling_matrix(scale * dst_shape / min(dst_shape)) @
            translation_matrix(-half))
  assert np.allclose(matrix[-1], [0.0, 0.0, 1.0])
  return matrix


def rotate_image_about_center(image: _NDArray,
                              angle: float,
                              new_shape: Any = None,
                              scale: float = 1.0,
                              num_rotations: int = 1,
                              **kwargs: Any) -> _NDArray:
  """Return a copy of `image` rotated about its center.

  Args:
    image: Source grid samples; the first two dimensions are spatial (ny, nx).
    angle: Angle in radians (positive from x to y axis) applied when mapping the source domain
      onto the destination domain.
    new_shape: Resolution (ny, nx) of the output grid; it defaults to `image.shape[:2]`.
    scale: Scaling factor applied when mapping the source domain onto the destination domain.
    num_rotations: Number of rotations (each by `angle`).  Successive resamplings are useful in
      analyzing the filtering quality.
    **kwargs: Additional parameters for `resample_affine`.
  """
  new_shape = image.shape[:2] if new_shape is None else np.asarray(new_shape)
  matrix = rotation_about_center_in_2d(image.shape[:2], angle, new_shape, scale=scale)
  for _ in range(num_rotations):
    image = resample_affine(image, new_shape, matrix[:-1], **kwargs)
  return image


def pil_image_resize(array: Any, shape: Sequence[int], filter: str) -> _NDArray:
  """Invoke `PIL.Image.resize` using the same parameters as `resize`."""
  array = np.asarray(array)
  assert 1 <= array.ndim <= 3
  assert np.issubdtype(array.dtype, np.floating)
  shape = tuple(shape)
  _check_eq(len(shape), 2 if array.ndim >= 2 else 1)
  if array.ndim == 1:
    return pil_image_resize(array[None], (1, *shape), filter=filter)
  import PIL.Image
  if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image
  pil_resample = {
      'impulse': PIL.Image.Resampling.NEAREST,
      'box': PIL.Image.Resampling.BOX,
      'triangle': PIL.Image.Resampling.BILINEAR,
      'hamming1': PIL.Image.Resampling.HAMMING,  # GeneralizedHammingFilter(1, a0=0.54)
      'cubic': PIL.Image.Resampling.BICUBIC,
      'lanczos3': PIL.Image.Resampling.LANCZOS,
  }[filter]
  if array.ndim == 2:
    return np.array(PIL.Image.fromarray(array).resize(
        shape[::-1], resample=pil_resample), dtype=array.dtype)
  return np.dstack(
      [np.array(PIL.Image.fromarray(channel).resize(
          shape[::-1], resample=pil_resample), dtype=array.dtype)
       for channel in np.moveaxis(array, -1, 0)])


def cv_resize(array: Any, shape: Sequence[int], filter: str) -> _NDArray:
  """Invoke `cv.resize` using the same parameters as `resize`."""
  array = np.asarray(array)
  assert 1 <= array.ndim <= 3
  shape = tuple(shape)
  _check_eq(len(shape), 2 if array.ndim >= 2 else 1)
  if array.ndim == 1:
    return cv_resize(array[None], (1, *shape), filter=filter)[0]
  import cv2 as cv
  interpolation = {
      'impulse': cv.INTER_NEAREST,  # Consider cv.INTER_NEAREST_EXACT?
      'triangle': cv.INTER_LINEAR_EXACT,  # Or just cv.INTER_LINEAR?
      'trapezoid': cv.INTER_AREA,
      'sharpcubic': cv.INTER_CUBIC,
      'lanczos4': cv.INTER_LANCZOS4,
  }[filter]
  return cv.resize(array, shape[::-1], interpolation=interpolation)


_TENSORFLOW_IMAGE_RESIZE_METHOD_FROM_FILTER = {
    'impulse': 'nearest',
    'trapezoid': 'area',
    'triangle': 'bilinear',
    'mitchell': 'mitchellcubic',
    'cubic': 'bicubic',
    'lanczos3': 'lanczos3',
    'lanczos5': 'lanczos5',
    # GaussianFilter(0.5): 'gaussian',  # radius_4 > desired_radius_3.
}

def tf_image_resize(array: Any, shape: Sequence[int], filter: str = 'lanczos3',
                    antialias: bool = True) -> _TensorflowTensor:
  """Invoke `tf.image.resize` using the same parameters as `resize`."""
  import tensorflow as tf
  array = tf.convert_to_tensor(array)
  assert 1 <= array.ndim <= 3
  shape = tuple(shape)
  _check_eq(len(shape), 2 if array.ndim >= 2 else 1)
  if array.ndim == 1:
    return tf_image_resize(array[None], (1, *shape), filter=filter, antialias=antialias)[0]
  if array.ndim == 2:
    return tf_image_resize(array[..., None], shape, filter=filter, antialias=antialias)[..., 0]
  method = _TENSORFLOW_IMAGE_RESIZE_METHOD_FROM_FILTER[filter]
  return tf.image.resize(array, shape, method=method, antialias=antialias)


_TORCH_INTERPOLATE_MODE_FROM_FILTER = {
    'impulse': 'nearest',
    'trapezoid': 'area',
    'triangle': 'bilinear',
    'sharpcubic': 'bicubic',
}

def torch_nn_resize(array: Any, shape: Sequence[int], filter: str,
                    antialias: bool = False) -> _TorchTensor:
  """Invoke `torch.nn.functional.interpolate` using the same parameters as `resize`."""
  import torch
  array = torch.as_tensor(array)
  assert 1 <= array.ndim <= 3
  shape = tuple(shape)
  _check_eq(len(shape), 2 if array.ndim >= 2 else 1)
  mode = _TORCH_INTERPOLATE_MODE_FROM_FILTER[filter]

  def local_resize(array: _TorchTensor) -> _TorchTensor:
    # For upsampling, BILINEAR antialias is same PSNR and slower,
    #  and BICUBIC antialias is worse PSNR and faster.
    # For downsampling, antialias improves PSNR for both BILINEAR and BICUBIC.
    align_corners = False if filter in ['linear', 'cubic'] else None
    return torch.nn.functional.interpolate(
        array, shape, mode=mode, align_corners=align_corners, antialias=antialias)

  if array.ndim == 1:
    shape = (1, *shape)
    return local_resize(array[None, None, None])[0, 0, 0]
  if array.ndim == 2:
    return local_resize(array[None, None])[0, 0]
  return local_resize(array.moveaxis(2, 0)[None])[0].moveaxis(0, 2)


def torchvision_resize(array: Any, shape: Sequence[int], filter: str,
                       antialias: Optional[bool] = None) -> _TorchTensor:
  """Invoke `torchvision.transforms.functional.resize` using the same parameters as `resize`."""
  # The results appear to be identical to `torch.nn.functional.interpolate` and slightly slower.
  import torch
  import torchvision
  array = torch.as_tensor(array)
  shape = tuple(shape)
  _check_eq(len(shape), 2 if array.ndim >= 2 else 1)
  torchvision_interpolation_from_filter = {
      'impulse': torchvision.transforms.InterpolationMode.NEAREST,
      'triangle': torchvision.transforms.InterpolationMode.BILINEAR,
      'sharpcubic': torchvision.transforms.InterpolationMode.BICUBIC,
      # Only the 3 modes above are supported when `array` is a torch.Tensor.
      # 'box': torchvision.transforms.InterpolationMode.BOX,
  }
  interpolation = torchvision_interpolation_from_filter[filter]

  def local_resize(array: _TorchTensor) -> _TorchTensor:
    return torchvision.transforms.functional.resize(
        array, shape, interpolation=interpolation, antialias=antialias)

  if array.ndim == 1:
    shape = (1, *shape)
    return local_resize(array[None, None])[0, 0]
  if array.ndim == 2:
    return local_resize(array[None])[0]
  return local_resize(array.moveaxis(2, 0)).moveaxis(0, 2)


# For Emacs:
# Local Variables: *
# fill-column: 100 *
# End: *
