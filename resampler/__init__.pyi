"""resampler: fast differentiable resizing and warping of arbitrary grids.

.. include:: ../README.md
-----
"""

from __future__ import annotations

import abc
import dataclasses
import typing
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

# pylint: disable=unused-argument, missing-function-docstring, missing-class-docstring
# pylint: disable=super-init-not-called

if typing.TYPE_CHECKING:
  _DType: TypeAlias = np.dtype[Any]
  _NDArray: TypeAlias = np.ndarray[Any, Any]
  _DTypeLike: TypeAlias = npt.DTypeLike
  _ArrayLike: TypeAlias = npt.ArrayLike
  #
  # Either:
  # (1) Strictly take all the dependencies on these large packages.
  import jax.experimental.sparse
  import jax.numpy
  import tensorflow as tf
  import torch

  _TensorflowTensor: TypeAlias = tf.Tensor
  _TorchTensor: TypeAlias = torch.Tensor
  _JaxArray: TypeAlias = jax.numpy.ndarray

  # (2) Bypass the dependencies to speed up mypy.
  # _TensorflowTensor: TypeAlias = Any
  # _TorchTensor: TypeAlias = Any
  # _JaxArray: TypeAlias = Any

  _Array = TypeVar('_Array', _NDArray, _TensorflowTensor, _TorchTensor, _JaxArray)
  _AnyArray = _NDArray | _TensorflowTensor | _TorchTensor | _JaxArray

else:
  # Create named types for use in the `pdoc` documentation.
  _DType = TypeVar('_DType')  # pylint: disable=invalid-name
  _NDArray = TypeVar('_NDArray')
  _DTypeLike = TypeVar('_DTypeLike')
  _ArrayLike = TypeVar('_ArrayLike')

  _TensorflowTensor = TypeVar('_TensorflowTensor')
  _TorchTensor = TypeVar('_TorchTensor')
  _JaxArray = TypeVar('_JaxArray')

  _Array = TypeVar('_Array')
  _AnyArray = TypeVar('_AnyArray')

__docformat__: str
__version_info__: tuple[int, ...]

ARRAYLIBS: list[str]

@dataclasses.dataclass(frozen=True)
class Gridtype(abc.ABC):
  name: str
  @abc.abstractmethod
  def min_size(self) -> int: ...
  @abc.abstractmethod
  def size_in_samples(self, size: int) -> int: ...
  @abc.abstractmethod
  def point_from_index(self, index: _NDArray, size: int) -> _NDArray: ...
  @abc.abstractmethod
  def index_from_point(self, point: _NDArray, size: int) -> _NDArray: ...
  @abc.abstractmethod
  def reflect(self, index: _NDArray, size: int) -> _NDArray: ...
  @abc.abstractmethod
  def wrap(self, index: _NDArray, size: int) -> _NDArray: ...
  @abc.abstractmethod
  def reflect_clamp(self, index: _NDArray, size: int) -> _NDArray: ...

class DualGridtype(Gridtype):
  def __init__(self) -> None: ...
  def min_size(self) -> int: ...
  def size_in_samples(self, size: int) -> int: ...
  def point_from_index(self, index: _NDArray, size: int) -> _NDArray: ...
  def index_from_point(self, point: _NDArray, size: int) -> _NDArray: ...
  def reflect(self, index: _NDArray, size: int) -> _NDArray: ...
  def wrap(self, index: _NDArray, size: int) -> _NDArray: ...
  def reflect_clamp(self, index: _NDArray, size: int) -> _NDArray: ...

class PrimalGridtype(Gridtype):
  def __init__(self) -> None: ...
  def min_size(self) -> int: ...
  def size_in_samples(self, size: int) -> int: ...
  def point_from_index(self, index: _NDArray, size: int) -> _NDArray: ...
  def index_from_point(self, point: _NDArray, size: int) -> _NDArray: ...
  def reflect(self, index: _NDArray, size: int) -> _NDArray: ...
  def wrap(self, index: _NDArray, size: int) -> _NDArray: ...
  def reflect_clamp(self, index: _NDArray, size: int) -> _NDArray: ...

GRIDTYPES: list[str]

@dataclasses.dataclass(frozen=True)
class RemapCoordinates(abc.ABC):
  @abc.abstractmethod
  def __call__(self, point: _NDArray) -> _NDArray: ...

class NoRemapCoordinates(RemapCoordinates):
  def __call__(self, point: _NDArray) -> _NDArray: ...

class MirrorRemapCoordinates(RemapCoordinates):
  def __call__(self, point: _NDArray) -> _NDArray: ...

class TileRemapCoordinates(RemapCoordinates):
  def __call__(self, point: _NDArray) -> _NDArray: ...

@dataclasses.dataclass(frozen=True)
class ExtendSamples(abc.ABC):
  uses_cval: bool = ...
  @abc.abstractmethod
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...
  # def __init__(self, uses_cval: bool) -> None: ...

class ReflectExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class WrapExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class ClampExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class ReflectClampExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class BorderExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class ValidExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class LinearExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

class QuadraticExtendSamples(ExtendSamples):
  def __call__(
      self, index: _NDArray, weight: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...

@dataclasses.dataclass(frozen=True)
class OverrideExteriorValue:
  boundary_antialiasing: bool = ...
  uses_cval: bool = ...
  def __call__(self, weight: _NDArray, point: _NDArray) -> None: ...
  def override_using_signed_distance(
      self, weight: _NDArray, point: _NDArray, signed_distance: _NDArray
  ) -> None: ...

class NoOverrideExteriorValue(OverrideExteriorValue):
  def __call__(self, weight: _NDArray, point: _NDArray) -> None: ...

class UnitDomainOverrideExteriorValue(OverrideExteriorValue):
  def __call__(self, weight: _NDArray, point: _NDArray) -> None: ...

class PlusMinusOneOverrideExteriorValue(OverrideExteriorValue):
  def __call__(self, weight: _NDArray, point: _NDArray) -> None: ...

@dataclasses.dataclass(frozen=True)
class Boundary:
  name: str = ...
  coord_remap: RemapCoordinates = ...
  extend_samples: ExtendSamples = ...
  override_value: OverrideExteriorValue = ...
  @property
  def uses_cval(self) -> bool: ...
  def preprocess_coordinates(self, point: _NDArray) -> _NDArray: ...
  def apply(
      self, index: _NDArray, weight: _NDArray, point: _NDArray, size: int, gridtype: Gridtype
  ) -> tuple[_NDArray, _NDArray]: ...
  def override_reconstruction(self, weight: _NDArray, point: _NDArray) -> None: ...

BOUNDARIES: list[str]

@dataclasses.dataclass(frozen=True)
class Filter(abc.ABC):
  name: str
  radius: float
  interpolating: bool = ...
  continuous: bool = ...
  partition_of_unity: bool = ...
  unit_integral: bool = ...
  requires_digital_filter: bool = ...
  @abc.abstractmethod
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class ImpulseFilter(Filter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class BoxFilter(Filter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class TrapezoidFilter(Filter):
  def __init__(self, *, radius: float | None = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class TriangleFilter(Filter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class CubicFilter(Filter):
  def __init__(self, *, b: float, c: float, name: str | None = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class CatmullRomFilter(CubicFilter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class MitchellFilter(CubicFilter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class SharpCubicFilter(CubicFilter):
  def __init__(self) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class LanczosFilter(Filter):
  def __init__(self, *, radius: int, sampled: bool = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class GeneralizedHammingFilter(Filter):
  a0: float
  def __init__(self, *, radius: int, a0: float) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class KaiserFilter(Filter):
  def __init__(self, *, radius: float, beta: float, sampled: bool = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class BsplineFilter(Filter):
  def __init__(self, *, degree: int) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class CardinalBsplineFilter(Filter):
  degree: int
  def __init__(self, *, degree: int, sampled: bool = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class OmomsFilter(Filter):
  degree: int
  def __init__(self, *, degree: int) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class GaussianFilter(Filter):
  DEFAULT_STANDARD_DEVIATION: float
  standard_deviation: float
  def __init__(self, *, standard_deviation: float = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

class NarrowBoxFilter(Filter):
  def __init__(self, *, radius: float = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> _NDArray: ...

FILTERS: list[str]

@dataclasses.dataclass(frozen=True)
class Gamma(abc.ABC):
  name: str
  @abc.abstractmethod
  def decode(self, array: _Array, dtype: _DTypeLike = ...) -> _Array: ...
  @abc.abstractmethod
  def encode(self, array: _Array, dtype: _DTypeLike) -> _Array: ...

class IdentityGamma(Gamma):
  def __init__(self) -> None: ...
  def decode(self, array: _Array, dtype: _DTypeLike = ...) -> _Array: ...
  def encode(self, array: _Array, dtype: _DTypeLike) -> _Array: ...

class PowerGamma(Gamma):
  power: float
  def __init__(self, power: float) -> None: ...
  def decode(self, array: _Array, dtype: _DTypeLike = ...) -> _Array: ...
  def encode(self, array: _Array, dtype: _DTypeLike) -> _Array: ...

class SrgbGamma(Gamma):
  def __init__(self) -> None: ...
  def decode(self, array: _Array, dtype: _DTypeLike = ...) -> _Array: ...
  def encode(self, array: _Array, dtype: _DTypeLike) -> _Array: ...

GAMMAS: list[str]

_DEFAULT_FILTER: str = 'lanczos3'

def resize(
    array: _Array,
    /,
    shape: Iterable[int],
    *,
    gridtype: str | Gridtype | None = None,
    src_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    dst_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    boundary: str | Boundary | Iterable[str | Boundary] = 'auto',
    cval: _ArrayLike = 0,
    filter: str | Filter | Iterable[str | Filter] = _DEFAULT_FILTER,
    prefilter: str | Filter | Iterable[str | Filter] | None = None,
    gamma: str | Gamma | None = None,
    src_gamma: str | Gamma | None = None,
    dst_gamma: str | Gamma | None = None,
    scale: float | Iterable[float] = 1.0,
    translate: float | Iterable[float] = 0.0,
    precision: _DTypeLike | None = None,
    dtype: _DTypeLike | None = None,
    dim_order: Iterable[int] | None = None,
    num_threads: int | Literal['auto'] = 'auto',
) -> _Array: ...

# def resize_in_arraylib(array: _NDArray, /, *args: Any, arraylib: str, **kwargs: Any) -> _NDArray: ...
def resize_in_arraylib(
    array: _NDArray, shape: Iterable[int], /, *args: Any, arraylib: str, **kwargs: Any
) -> _NDArray: ...
def resize_in_numpy(
    array: _NDArray, shape: Iterable[int], /, *args: Any, **kwargs: Any
) -> _NDArray: ...
def resize_in_tensorflow(
    array: _NDArray, shape: Iterable[int], /, *args: Any, **kwargs: Any
) -> _NDArray: ...
def resize_in_torch(
    array: _NDArray, shape: Iterable[int], /, *args: Any, **kwargs: Any
) -> _NDArray: ...
def resize_in_jax(
    array: _NDArray, shape: Iterable[int], /, *args: Any, **kwargs: Any
) -> _NDArray: ...
def jaxjit_resize(array: _Array, /, *args: Any, **kwargs: Any) -> _Array: ...

#
def uniform_resize(
    array: _Array,
    /,
    shape: Iterable[int],
    *,
    object_fit: Literal['contain', 'cover'] = 'contain',
    gridtype: str | Gridtype | None = None,
    src_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    dst_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    boundary: str | Boundary | Iterable[str | Boundary] = 'border',  # Instead of 'auto' default.
    scale: float | Iterable[float] = 1.0,
    translate: float | Iterable[float] = 0.0,
    **kwargs: Any,
) -> _Array: ...

#
def resample(
    array: _Array,
    /,
    coords: _ArrayLike,
    *,
    gridtype: str | Gridtype | Iterable[str | Gridtype] = 'dual',
    boundary: str | Boundary | Iterable[str | Boundary] = 'auto',
    cval: _ArrayLike = 0,
    filter: str | Filter | Iterable[str | Filter] = _DEFAULT_FILTER,
    prefilter: str | Filter | Iterable[str | Filter] | None = None,
    gamma: str | Gamma | None = None,
    src_gamma: str | Gamma | None = None,
    dst_gamma: str | Gamma | None = None,
    jacobian: _ArrayLike | None = None,
    precision: _DTypeLike | None = None,
    dtype: _DTypeLike | None = None,
    max_block_size: int = 40_000,
    debug: bool = False,
) -> _Array: ...
def resample_affine(
    array: _Array,
    /,
    shape: Iterable[int],
    matrix: _ArrayLike,
    *,
    gridtype: str | Gridtype | None = None,
    src_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    dst_gridtype: str | Gridtype | Iterable[str | Gridtype] | None = None,
    filter: str | Filter | Iterable[str | Filter] = _DEFAULT_FILTER,
    prefilter: str | Filter | Iterable[str | Filter] | None = None,
    precision: _DTypeLike | None = None,
    dtype: _DTypeLike | None = None,
    **kwargs: Any,
) -> _Array: ...

#
def rotation_about_center_in_2d(
    src_shape: _ArrayLike,
    /,
    angle: float,
    *,
    new_shape: _ArrayLike | None = None,
    scale: float = 1.0,
) -> _NDArray: ...
def rotate_image_about_center(
    image: _NDArray,
    /,
    angle: float,
    *,
    new_shape: _ArrayLike | None = None,
    scale: float = 1.0,
    num_rotations: int = 1,
    **kwargs: Any,
) -> _NDArray: ...

#
def _pil_image_resize(array: _ArrayLike, /, shape: Iterable[int], *, filter: str) -> _NDArray: ...
def _cv_resize(array: _ArrayLike, /, shape: Iterable[int], *, filter: str) -> _NDArray: ...
def _scipy_ndimage_resize(
    array: _ArrayLike,
    /,
    shape: Iterable[int],
    *,
    filter: str,
    boundary: str = 'reflect',
    cval: float = 0.0,
) -> _NDArray: ...
def _skimage_transform_resize(
    array: _ArrayLike,
    /,
    shape: Iterable[int],
    *,
    filter: str,
    boundary: str = 'reflect',
    cval: float = 0.0,
) -> _NDArray: ...
def _tf_image_resize(
    array: _ArrayLike, /, shape: Iterable[int], *, filter: str, antialias: bool = True
) -> _TensorflowTensor: ...
def _torch_nn_resize(
    array: _ArrayLike, /, shape: Iterable[int], *, filter: str, antialias: bool = False
) -> _TorchTensor: ...
def _jax_image_resize(
    array: _ArrayLike,
    /,
    shape: Iterable[int],
    *,
    filter: str,
    scale: float | Iterable[float] = 1.0,
    translate: float | Iterable[float] = 0.0,
) -> _JaxArray: ...

#
def _check_eq(a: Any, b: Any, /) -> None: ...
def _real_precision(dtype: _DTypeLike, /) -> _DType: ...
def _complex_precision(dtype: _DTypeLike, /) -> _DType: ...
def _get_precision(
    precision: _DTypeLike | None, dtypes: list[_DType], weight_dtypes: list[_DType], /
) -> _DType: ...
def _sinc(x: _ArrayLike, /) -> _NDArray: ...
def _is_symmetric(matrix: _NDArray, /, tol: float = ...) -> bool: ...
def _cache_sampled_1d_function(
    xmin: float, xmax: float, *, num_samples: int = ..., enable: bool = ...
) -> Callable[[Callable[[_ArrayLike], _NDArray]], Callable[[_ArrayLike], _NDArray]]: ...

#
def _as_arr(array: _AnyArray, /) -> Any: ...  # -> _Arraylib[_Array]
def _arr_arraylib(array: _AnyArray, /) -> str: ...
def _arr_numpy(array: _AnyArray, /) -> _NDArray: ...
def _arr_dtype(array: _AnyArray, /) -> _DType: ...
def _arr_astype(array: _Array, dtype: _DTypeLike, /) -> _Array: ...
def _arr_reshape(array: _Array, shape: tuple[int, ...], /) -> _Array: ...
def _arr_possibly_make_contiguous(array: _Array, /) -> _Array: ...
def _arr_clip(array: _Array, low: Any, high: Any, /, dtype: _DTypeLike | None = ...) -> _Array: ...
def _arr_square(array: _Array, /) -> _Array: ...
def _arr_sqrt(array: _Array, /) -> _Array: ...
def _arr_getitem(array: _Array, indices: _Array, /) -> _Array: ...
def _arr_where(condition: _Array, if_true: Any, if_false: Any, /) -> _Array: ...
def _arr_transpose(array: _Array, axes: Sequence[int], /) -> _Array: ...
def _arr_best_dims_order_for_resize(
    array: _AnyArray, dst_shape: tuple[int, ...], /
) -> list[int]: ...
def _arr_matmul_sparse_dense(
    sparse: Any, dense: _Array, /, *, num_threads: int | Literal['auto'] = 'auto'
) -> _Array: ...
def _arr_concatenate(arrays: Sequence[_Array], axis: int, /) -> _Array: ...
def _arr_einsum(subscripts: str, /, *operands: _Array) -> _Array: ...
def _arr_swapaxes(array: _Array, axis1: int, axis2: int, /) -> _Array: ...
def _arr_moveaxis(array: _Array, source: int, destination: int, /) -> _Array: ...

#
def _make_sparse_matrix(
    data: _NDArray, row_ind: _NDArray, col_ind: _NDArray, shape: tuple[int, int], arraylib: str, /
) -> Any: ...
def _make_array(array: _ArrayLike, arraylib: str, /) -> Any: ...
def _block_shape_with_min_size(
    shape: tuple[int, ...], min_size: int, compact: bool = ...
) -> tuple[int, ...]: ...
def _array_split(array: _Array, axis: int, num_sections: int) -> list[_Array]: ...
def _split_array_into_blocks(
    array: _AnyArray, block_shape: Sequence[int], start_axis: int = ...
) -> Any: ...
def _map_function_over_blocks(blocks: Any, func: Callable[[Any], Any]) -> Any: ...
def _merge_array_from_blocks(blocks: Any, axis: int = ...) -> Any: ...
def _get_gridtype(gridtype: str | Gridtype) -> Gridtype: ...
def _get_gridtypes(
    gridtype: str | Gridtype | None,
    src_gridtype: str | Gridtype | Iterable[str | Gridtype] | None,
    dst_gridtype: str | Gridtype | Iterable[str | Gridtype] | None,
    src_ndim: int,
    dst_ndim: int,
) -> tuple[list[Gridtype], list[Gridtype]]: ...
def _get_boundary(boundary: str | Boundary, /) -> Boundary: ...
def _get_filter(filter: str | Filter, /) -> Filter: ...
def _to_float_01(array: _Array, /, dtype: _DTypeLike) -> _Array: ...
def _from_float(array: _Array, /, dtype: _DTypeLike) -> _Array: ...
def _get_gamma(gamma: str | Gamma, /) -> Gamma: ...
def _get_src_dst_gamma(
    gamma: str | Gamma | None,
    src_gamma: str | Gamma | None,
    dst_gamma: str | Gamma | None,
    src_dtype: _DType,
    dst_dtype: _DType,
) -> tuple[Gamma, Gamma]: ...
def _create_resize_matrix(
    src_size: int,
    dst_size: int,
    src_gridtype: Gridtype,
    dst_gridtype: Gridtype,
    boundary: Boundary,
    filter: Filter,
    prefilter: Filter | None = ...,
    scale: float = ...,
    translate: float = ...,
    dtype: _DTypeLike = ...,
    arraylib: str = ...,
) -> tuple[Any, _AnyArray | None]: ...
def _apply_digital_filter_1d(
    array: _Array,
    gridtype: Gridtype,
    boundary: Boundary,
    cval: _ArrayLike,
    filter: Filter,
    /,
    *,
    axis: int = ...,
) -> _Array: ...
def _apply_digital_filter_1d_numpy(
    array: _NDArray,
    gridtype: Gridtype,
    boundary: Boundary,
    cval: _ArrayLike,
    filter: Filter,
    axis: int,
    compute_backward: bool,
    /,
) -> _NDArray: ...
def _resize_possibly_in_arraylib(
    array: _AnyArray, /, *args: Any, arraylib: str, **kwargs: Any
) -> _AnyArray: ...
def _create_jaxjit_resize() -> Callable[..., Any]: ...
def _resize_using_resample(
    array: _Array,
    /,
    shape: Iterable[int],
    *,
    scale: _ArrayLike = ...,
    translate: _ArrayLike = ...,
    filter: str | Filter | Iterable[str | Filter] = ...,
    fallback: bool = ...,
    **kwargs: Any,
) -> _Array: ...

class _DownsampleIn2dUsingBoxFilter:
  def __init__(self) -> None: ...
  def __call__(self, array: _NDArray, shape: tuple[int, int]) -> _NDArray: ...

_downsample_in_2d_using_box_filter: _DownsampleIn2dUsingBoxFilter

def _numba_serial_csr_dense_mult(
    indptr: _NDArray,
    indices: _NDArray,
    data: _NDArray,
    src: _NDArray,
    dst: _NDArray,
) -> None: ...
def _numba_parallel_csr_dense_mult(
    indptr: _NDArray,
    indices: _NDArray,
    data: _NDArray,
    src: _NDArray,
    dst: _NDArray,
) -> None: ...

_TENSORFLOW_IMAGE_RESIZE_METHOD_FROM_FILTER: dict[str, str]
_TORCH_INTERPOLATE_MODE_FROM_FILTER: dict[str, str]
_OFTUSED_BOUNDARIES: list[str]
_DICT_GRIDTYPES: dict[str, Gridtype]
_DICT_BOUNDARIES: dict[str, Boundary]
_DICT_FILTERS: dict[str, Filter]
_DICT_GAMMAS: dict[str, Gamma]
_RESIZERS: dict[str, Callable[..., _AnyArray]]

def _original_resize(
    array: _NDArray, shape: Iterable[int], *args: Any, **kwargs: Any
) -> _NDArray: ...
def _find_closest_filter(filter: str, resizer: Callable[..., Any]) -> str: ...
