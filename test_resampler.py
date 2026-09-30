#!/usr/bin/env python3
# -*- fill-column: 100; -*-
"""Tests for package resampler.

Run them using `pytest` in the repository root (whose pyproject.toml also enables the doctests)
or using `python3 test_resampler.py`.
"""

import functools
import itertools
import math
import os
import unittest
import warnings
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import numpy.typing
import scipy.interpolate
import scipy.sparse

import resampler

_ArrayLike: TypeAlias = numpy.typing.ArrayLike
_NDArray: TypeAlias = numpy.typing.NDArray[Any]

# pylint: disable=protected-access, missing-function-docstring, too-many-public-methods

# Silence "WARNING:absl:No GPU/TPU found, falling back to CPU".
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


def enable_jax_float64() -> None:
  """Enable use of double-precision float in Jax; this only works at startup."""
  import jax  # ("import jax.config" is disallowed.)

  jax.config.update('jax_enable_x64', True)


def _check_eq(a: Any, b: Any) -> None:
  """If the two values or arrays are not equal, raise an exception with a useful message."""
  are_equal = np.all(a == b) if isinstance(a, np.ndarray) else a == b
  if not are_equal:
    raise AssertionError(f'{a!r} == {b!r}')


class TestResampler(unittest.TestCase):
  """Test class for resampler package."""

  @classmethod
  def setUpClass(cls: type) -> None:
    if 'jax' in resampler.ARRAYLIBS:
      enable_jax_float64()
    # Silence the warning in package flatbuffers.
    warnings.filterwarnings('ignore', message='.*the imp module is deprecated')

  def test_resize_on_array(self) -> None:
    array = np.array([3.0, 5.0, 8.0, 7.0])
    expected = np.array([2.84536097, 3.6902174, 5.58573019, 7.77282572, 7.8097826, 6.79608312])
    np.testing.assert_allclose(resampler.resize(array, (6,)), expected)
    for arraylib in resampler.ARRAYLIBS:
      with self.subTest(arraylib=arraylib):
        new = resampler._original_resize(resampler._make_array(array, arraylib), (6,))
        _check_eq(resampler._arr_arraylib(new), arraylib)
        np.testing.assert_allclose(resampler._arr_numpy(new), expected)

  def test_precision(self) -> None:
    _check_eq(resampler._real_precision(np.dtype(np.float32)), np.float32)
    _check_eq(resampler._real_precision(np.dtype(np.float64)), np.float64)
    _check_eq(resampler._real_precision(np.dtype(np.complex64)), np.float32)
    _check_eq(resampler._real_precision(np.dtype(np.complex128)), np.float64)

  def test_get_precision(self) -> None:
    _check_eq(
        resampler._get_precision(None, [np.dtype(np.complex64)], [np.dtype(np.float64)]),
        np.complex128,
    )

  def test_cached_sampling(self) -> None:
    radius = 2.0

    def create_scipy_interpolant(
        func: Callable[[_ArrayLike], _NDArray], xmin: float, xmax: float, num_samples: int = 3_600
    ) -> Callable[[_NDArray], _NDArray]:
      samples_x = np.linspace(xmin, xmax, num_samples + 1, dtype=np.float32)
      samples_func = func(samples_x)
      assert np.all(samples_func[[0, -1]] == 0.0)
      interpolator: Callable[[_NDArray], _NDArray] = scipy.interpolate.interp1d(
          samples_x, samples_func, kind='linear', bounds_error=False, fill_value=0
      )
      return interpolator

    def func(x: _ArrayLike) -> _NDArray:  # Lanczos kernel
      x = np.abs(x)
      return np.where(x < radius, resampler._sinc(x) * resampler._sinc(x / radius), 0.0)

    @resampler._cache_sampled_1d_function(xmin=-radius, xmax=radius)
    def func2(x: _ArrayLike) -> _NDArray:
      return func(x)

    scipy_interp = create_scipy_interpolant(func, -radius, radius)

    shape = 2, 8_000
    rng = np.random.default_rng(1)
    array = rng.random(shape, np.float32) * 2 * radius - radius
    result = {'expected': func(array), 'scipy': scipy_interp(array), 'obtained': func2(array)}

    assert all(a.dtype == np.float32 for a in result.values())
    assert all(a.shape == shape for a in result.values())
    assert np.allclose(result['scipy'], result['expected'], rtol=0, atol=1e-6)
    assert np.allclose(result['obtained'], result['expected'], rtol=0, atol=1e-6)

  @unittest.skipIf(not resampler._USING_NUMBA, 'The box-filter downsampling requires numba.')
  def test_downsample_in_2d_using_box_filter(self) -> None:
    for shape in [(6, 6), (4, 4)]:
      for ch in [1, 2, 3, 4]:
        array1 = np.ones((*shape, ch), np.float32)
        new1 = resampler._downsample_in_2d_using_box_filter(array1, (2, 2))
        _check_eq(new1.shape, (2, 2, ch))
        assert np.allclose(new1, 1.0)

    for shape in [(6, 6), (4, 4)]:
      array2 = np.ones(shape, np.float32)
      new2 = resampler._downsample_in_2d_using_box_filter(array2, (2, 2))
      _check_eq(new2.shape, (2, 2))
      assert np.allclose(new2, 1.0)

  def test_block_shape_with_min_size(self) -> None:
    for compact in [True, False]:
      with self.subTest(compact=compact):
        shape = 2, 3, 4
        for min_size in range(1, math.prod(shape) + 1):
          block_shape = resampler._block_shape_with_min_size(shape, min_size, compact=compact)
          assert np.all(np.array(block_shape) >= 1)
          assert np.all(np.array(block_shape) <= shape)
          assert min_size <= math.prod(block_shape) <= math.prod(shape)

  def test_split_2d(self) -> None:
    numpy_array = np.random.default_rng(1).choice([1, 2, 3, 4], (5, 8))
    for arraylib in resampler.ARRAYLIBS:
      array = resampler._make_array(numpy_array, arraylib)
      blocks = resampler._split_array_into_blocks(array, [2, 3])
      blocks = resampler._map_function_over_blocks(blocks, lambda x: 2 * x)
      new = resampler._merge_array_from_blocks(blocks)
      _check_eq(resampler._arr_arraylib(new), arraylib)
      _check_eq(np.sum(resampler._map_function_over_blocks(blocks, lambda _: 1)), 9)
      _check_eq(resampler._arr_numpy(new), 2 * numpy_array)

  def test_split_3d(self) -> None:
    shape = 4, 3, 2
    numpy_array = np.random.default_rng(1).choice([1, 2, 3, 4], shape)

    for arraylib in resampler.ARRAYLIBS:
      array = resampler._make_array(numpy_array, arraylib)
      for min_size in range(1, math.prod(shape) + 1):
        block_shape = resampler._block_shape_with_min_size(shape, min_size)
        blocks = resampler._split_array_into_blocks(array, block_shape)
        blocks = resampler._map_function_over_blocks(blocks, lambda x: x**2)
        new = resampler._merge_array_from_blocks(blocks)
        _check_eq(resampler._arr_arraylib(new), arraylib)
        _check_eq(resampler._arr_numpy(new), numpy_array**2)

        def check_block_shape(block: _NDArray) -> None:
          assert np.all(np.array(block.shape) >= 1)
          assert np.all(np.array(block.shape) <= shape)

        resampler._map_function_over_blocks(blocks, check_block_shape)

  def test_split_prefix_dims(self) -> None:
    shape = 2, 3, 2
    array = np.arange(math.prod(shape)).reshape(shape)

    for min_size in range(1, math.prod(shape[:2]) + 1):
      block_shape = resampler._block_shape_with_min_size(shape[:2], min_size)
      blocks = resampler._split_array_into_blocks(array, block_shape)

      new_blocks = resampler._map_function_over_blocks(blocks, lambda x: x**2)
      new = resampler._merge_array_from_blocks(new_blocks)
      _check_eq(new, array**2)

      new_blocks = resampler._map_function_over_blocks(blocks, lambda x: x.sum(axis=-1))
      new = resampler._merge_array_from_blocks(new_blocks)
      _check_eq(new, array.sum(axis=-1))

  def test_linear_boundary(self) -> None:
    index = np.array([[-3], [-2], [-1], [0], [1], [2], [3], [2], [3]])
    weight = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0]])
    size = 2
    index, weight = resampler.LinearExtendSamples()(index, weight, size, resampler.DualGridtype())
    expected_weight = [
        [0, 4, -3, 0, 0],
        [0, 3, -2, 0, 0],
        [0, 2, -1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, -1, 2],
        [0, 0, 0, -2, 3],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert np.allclose(weight, expected_weight)
    expected_index = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert np.all(index == expected_index)

  def test_gamma_roundtrip_uint(self) -> None:
    dtypes = 'uint8 uint16 uint32'.split()
    for config in itertools.product(resampler.ARRAYLIBS, resampler.GAMMAS, dtypes):
      arraylib, gamma_name, dtype = config
      gamma = resampler._get_gamma(gamma_name)
      if arraylib == 'torch' and dtype in ['uint16', 'uint32']:
        continue  # "The only supported types are: ..., int64, int32, int16, int8, uint8, and bool."
      with self.subTest(config=config):
        int_max = np.iinfo(dtype).max
        precision = 'float32' if np.iinfo(dtype).bits < 32 else 'float64'
        values = list(range(256)) + list(range(int_max - 255, int_max)) + [int_max]
        array_numpy = np.array(values, dtype)
        array = resampler._make_array(array_numpy, arraylib)
        decoded = gamma.decode(array, np.dtype(precision))
        _check_eq(resampler._arr_dtype(decoded), precision)
        decoded_numpy = resampler._arr_numpy(decoded)
        assert decoded_numpy.min() >= 0.0 and decoded_numpy.max() <= 1.0
        encoded = gamma.encode(decoded, dtype)
        _check_eq(resampler._arr_dtype(encoded), dtype)
        encoded_numpy = resampler._arr_numpy(encoded)
        _check_eq(encoded_numpy, array_numpy)

  def test_gamma_roundtrip_float(self) -> None:
    dtypes = 'float32 float64'.split()
    precisions = 'float32 float64'.split()
    for config in itertools.product(resampler.ARRAYLIBS, resampler.GAMMAS, dtypes, precisions):
      arraylib, gamma_name, dtype, precision = config
      with self.subTest(config=config):
        gamma = resampler._get_gamma(gamma_name)
        array_numpy = np.linspace(0.0, 1.0, 100, dtype=dtype)
        array = resampler._make_array(array_numpy, arraylib)
        decoded = gamma.decode(array, np.dtype(precision))
        _check_eq(resampler._arr_dtype(decoded), precision)
        encoded = gamma.encode(decoded, dtype)
        _check_eq(resampler._arr_dtype(encoded), dtype)
        assert np.allclose(resampler._arr_numpy(encoded), array_numpy)

  def test_create_resize_matrix_for_trapezoid_filter(self) -> None:
    filter = resampler.TrapezoidFilter()
    for src_size, dst_size in [(6, 2), (7, 3), (7, 6), (14, 13), (3, 6), (3, 12), (3, 11), (3, 16)]:
      with self.subTest(src_size=src_size, dst_size=dst_size):
        resize_matrix, unused_cval_weight = resampler._create_resize_matrix(
            src_size,
            dst_size,
            src_gridtype=resampler.DualGridtype(),
            dst_gridtype=resampler.DualGridtype(),
            boundary=resampler._get_boundary('reflect'),
            filter=filter,
        )
        resize_matrix = resize_matrix.toarray()
        assert resize_matrix.sum(axis=0).var() < 1e-10
        assert resize_matrix.sum(axis=1).var() < 1e-10

  def test_sparse_csr_matrix_duplicate_entries_are_summed(self) -> None:
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 0])
    data = np.array([1, 2, 3, 4, 5, 3])
    new = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    _check_eq(new, [[1, 0, 2], [0, 0, 3], [7, 5, 0]])

  def test_that_resize_matrices_are_equal_across_arraylib(self) -> None:
    src_sizes = range(1, 6)
    dst_sizes = range(1, 6)
    for config in itertools.product(src_sizes, dst_sizes):
      src_size, dst_size = config
      with self.subTest(config=config):

        def resize_matrix(arraylib: str) -> Any:
          return resampler._create_resize_matrix(
              src_size,  # noqa: B023
              dst_size,  # noqa: B023
              src_gridtype=resampler.DualGridtype(),
              dst_gridtype=resampler.DualGridtype(),
              boundary=resampler._get_boundary('reflect'),
              filter=resampler._get_filter('lanczos3'),
              translate=0.8,
              dtype=np.float32,
              arraylib=arraylib,
          )[0]

        results: dict[str, _NDArray] = {}
        for arraylib in resampler.ARRAYLIBS:
          sparse_matrix = resize_matrix(arraylib)
          match arraylib:
            case 'numpy':
              result = sparse_matrix.toarray()
            case 'torch':
              result = sparse_matrix.to_dense().numpy()
            case 'jax':
              result = np.array(sparse_matrix.todense())
            case _:
              raise AssertionError
          results[arraylib] = result
        for arraylib in resampler.ARRAYLIBS:
          assert np.allclose(results[arraylib], results['numpy'])

  def test_that_resize_combinations_are_affine(self) -> None:
    dst_sizes = 1, 2, 3, 4, 9, 20, 21, 22, 31
    for config in itertools.product(resampler.BOUNDARIES, dst_sizes):
      boundary, dst_size = config
      with self.subTest(config=config):
        resize_matrix, cval_weight = resampler._create_resize_matrix(
            21,
            dst_size,
            src_gridtype=resampler.DualGridtype(),
            dst_gridtype=resampler.DualGridtype(),
            boundary=resampler._get_boundary(boundary),
            filter=resampler.TriangleFilter(),
            scale=0.5,
            translate=0.3,
        )
        row_sum = np.asarray(resize_matrix.sum(axis=1)).ravel()
        if cval_weight is not None:
          row_sum += np.asarray(cval_weight)
        assert np.allclose(row_sum, 1.0, rtol=0, atol=1e-6), (resize_matrix.todense(), row_sum)

  def test_linear_precision_of_1d_primal_upsampling(self) -> None:
    array = np.arange(7.0)
    new = resampler.resize(array, (13,), gridtype='primal', filter='triangle')
    with np.printoptions(linewidth=300):  # (For the message of a failed check.)
      _check_eq(new, np.arange(13) / 2)

  def test_linear_precision_of_2d_primal_upsampling(self) -> None:
    shape = 3, 5
    new_shape = 5, 9
    array = np.moveaxis(np.indices(shape, np.float32), 0, -1) @ [10, 1]
    new = resampler.resize(array, new_shape, gridtype='primal', filter='triangle')
    with np.printoptions(linewidth=300):  # (For the message of a failed check.)
      expected = np.moveaxis(np.indices(new_shape, np.float32), 0, -1) @ [10, 1] / 2
      _check_eq(new, expected)

  def test_resize_of_complex_value_type(self) -> None:
    for arraylib in resampler.ARRAYLIBS:
      array = resampler._make_array([1 + 2j, 3 + 6j], arraylib)
      new = resampler._original_resize(array, (4,), filter='triangle')
      assert np.allclose(new, [1 + 2j, 1.5 + 3j, 2.5 + 5j, 3 + 6j])

  def test_resize_of_integer_type(self) -> None:
    array = np.array([1, 6])
    new = resampler.resize(array, (4,), filter='triangle')
    assert np.allclose(new, [1, 2, 5, 6])

  def test_apply_digital_filter_1d(self) -> None:
    cval = -10.0
    shape = 7, 8
    original = np.arange(math.prod(shape), dtype=np.float32).reshape(shape) + 10
    array1 = original.copy()
    filters = 'cardinal3 cardinal5'.split()
    for config in itertools.product(resampler.GRIDTYPES, resampler.BOUNDARIES, filters):
      gridtype, boundary, filter = config
      with self.subTest(config=config):
        if gridtype == 'primal' and boundary in ('wrap', 'tile'):
          continue  # Last value on each dimension is ignored and so will not match.
        array2 = array1
        for dim in range(array2.ndim):
          array2 = resampler._apply_digital_filter_1d(
              array2,
              resampler._get_gridtype(gridtype),
              resampler._get_boundary(boundary),
              cval,
              resampler._get_filter(filter),
              axis=dim,
          )
        bspline = resampler.BsplineFilter(degree=int(filter[-1:]))
        array3 = resampler.resize(
            array2, array2.shape, gridtype=gridtype, boundary=boundary, cval=cval, filter=bspline
        )
        assert np.allclose(array3, original)

  def test_resample_small_arrays(self) -> None:
    shape = 2, 3
    new_shape = 3, 4
    for arraylib in resampler.ARRAYLIBS:
      with self.subTest(arraylib=arraylib):
        array = np.arange(math.prod(shape) * 3, dtype=np.float32).reshape(shape + (3,))
        coords = np.moveaxis(np.indices(new_shape) + 0.5, 0, -1) / new_shape
        array = resampler._make_array(array, arraylib)
        upsampled = resampler.resample(array, coords)
        _check_eq(upsampled.shape, (*new_shape, 3))
        coords = np.moveaxis(np.indices(shape) + 0.5, 0, -1) / shape
        downsampled = resampler.resample(upsampled, coords)
        difference = resampler._arr_numpy(array) - resampler._arr_numpy(downsampled)
        rms = np.sqrt(np.mean(np.square(difference))).item()
        assert 0.07 <= rms <= 0.08, rms

  def test_identity_resampling_with_many_boundary_rules(self) -> None:
    filter = resampler.LanczosFilter(radius=5, sampled=False)
    for boundary in resampler.BOUNDARIES:
      with self.subTest(boundary=boundary):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        coords = (np.moveaxis(np.indices(array.shape), 0, -1) + 0.5) / array.shape
        new_array = resampler.resample(array, coords, boundary=boundary, cval=10000, filter=filter)
        assert np.allclose(new_array, array), boundary

  def test_identity_resampling(self) -> None:
    shape = 3, 2, 5
    array = np.random.default_rng(1).random(shape)
    coords = (np.moveaxis(np.indices(array.shape), 0, -1) + 0.5) / array.shape
    new = resampler.resample(array, coords)
    assert np.allclose(new, array, rtol=0, atol=1e-6)
    new = resampler.resample(array, coords, filter=resampler.LanczosFilter(radius=3, sampled=False))
    assert np.allclose(new, array)

  def test_resample_of_complex_value_type(self) -> None:
    array = np.array([1 + 2j, 3 + 6j])
    new = resampler.resample(array, (0.125, 0.375, 0.625, 0.875), filter='triangle')
    assert np.allclose(new, [1 + 2j, 1.5 + 3j, 2.5 + 5j, 3 + 6j])

  def test_resample_of_integer_type(self) -> None:
    array = np.array([1, 6])
    new = resampler.resample(array, (0.125, 0.375, 0.625, 0.875), filter='triangle')
    assert np.allclose(new, [1, 2, 5, 6])

  def test_resample_using_coords_of_various_shapes(self) -> None:
    for lst in [
        8,
        [7],
        [0, 1, 6, 6],
        [[0, 1], [10, 16]],
        [[0], [1], [6], [6]],
    ]:
      with self.subTest(lst=lst):
        array = np.array(lst, np.float64)
        for shape in [(), (1,), (2,), (1, 1), (1, 2), (3, 1), (2, 2)]:
          coords: _NDArray = np.full(shape, 0.4)
          try:
            new = resampler.resample(array, coords, filter='triangle', dtype=np.float32).tolist()
          except ValueError:
            new = None
          # print(f'{array.tolist()!s:30} {coords.shape!s:8} {new!s}')
          _check_eq(new is None, coords.ndim >= 2 and coords.shape[-1] > max(array.ndim, 1))

  def test_resize_using_resample(self) -> None:
    shape = 3, 2, 5
    new_shape = 4, 2, 7
    step = 37
    assert np.all(np.array(shape) <= new_shape)
    array = np.random.default_rng(1).random(shape)
    scale = 1.1
    translate = -0.4, -0.03, 0.4
    gammas = 'identity power2'.split()  # Sublist of resampler.GAMMAS.
    sequences = [resampler.GRIDTYPES, resampler.BOUNDARIES, resampler.FILTERS, gammas]
    assert step == 1 or all(len(sequence) % step != 0 for sequence in sequences)
    configs = itertools.product(*sequences)  # len(configs) = math.prod([2, 12, 19, 2]) = 912.
    for config in itertools.islice(configs, 0, None, step):
      gridtype, boundary, filter, gamma = config
      with self.subTest(config=config):
        kwargs: Any = dict(gridtype=gridtype, boundary=boundary, filter=filter)
        kwargs |= dict(gamma=gamma, scale=scale, translate=translate)
        expected = resampler._original_resize(array, new_shape, **kwargs)
        new_array = resampler._resize_using_resample(array, new_shape, **kwargs)
        assert np.allclose(new_array, expected, rtol=0, atol=1e-7)

  def test_resize_using_resample_of_complex_value_type(self) -> None:
    array = np.array([1 + 2j, 3 + 6j])
    new = resampler._resize_using_resample(array, (4,), filter='triangle')
    assert np.allclose(new, [1 + 2j, 1.5 + 3j, 2.5 + 5j, 3 + 6j])

  def test_resizers_produce_correct_shape(self) -> None:
    configs: list[tuple[Callable[..., Any], str]] = [(resampler.resize, 'lanczos3')]
    for arraylib in resampler.ARRAYLIBS:
      resizer0 = functools.partial(resampler.resize_in_arraylib, arraylib=arraylib)
      configs.append((resizer0, 'lanczos3'))
    configs.append((resampler._pil_image_resize, 'lanczos3'))
    configs.append((resampler._cv_resize, 'lanczos4'))
    configs.append((resampler._scipy_ndimage_resize, 'cardinal3'))
    configs.append((resampler._skimage_transform_resize, 'cardinal3'))
    configs.append((resampler._torch_nn_resize, 'sharpcubic'))
    configs.append((resampler._jax_image_resize, 'lanczos3'))
    for config in configs:
      resizer, filter = config
      is_external = not isinstance(resizer, functools.partial) and resizer != resampler.resize
      if is_external and resizer not in resampler._RESIZERS.values():
        continue  # Skip if the package is not installed.
      # The older scipy.ndimage (e.g., 1.7.2 in the minimum-versions test) deviates by ~1.4e-4.
      atol = 1e-3 if resizer == resampler._scipy_ndimage_resize else 1e-6
      with self.subTest(config=config):
        for src_shape, shape in [((11,), (13,)), ((8, 8), (5, 20)), ((9, 8, 3), (13, 7))]:
          new = np.asarray(resizer(np.ones(src_shape), shape, filter=filter))
          _check_eq(new.shape, shape + src_shape[len(shape) :])
          assert np.allclose(new, 1.0, rtol=0, atol=atol), new

  def test_boundary_names_match_their_keys(self) -> None:
    for name in resampler.BOUNDARIES:
      _check_eq(resampler._get_boundary(name).name, name)

  def test_generalized_hamming_filters_with_different_a0_differ(self) -> None:
    filter1 = resampler.GeneralizedHammingFilter(radius=3, a0=0.5)
    filter2 = resampler.GeneralizedHammingFilter(radius=3, a0=0.9)
    assert filter1 != filter2 and hash(filter1) != hash(filter2)

  def test_bspline_filter_of_degree_0_is_discontinuous(self) -> None:
    assert not resampler.BsplineFilter(degree=0).continuous
    assert resampler.BsplineFilter(degree=1).continuous

  def test_resample_cval_with_coords_ndim_differing_from_grid_ndim(self) -> None:
    image = np.ones((4, 4))
    coords = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]])  # Points along a line.
    for boundary in ['border', 'constant']:
      with self.subTest(boundary=boundary):
        kwargs: Any = dict(boundary=boundary, cval=5.0, filter='triangle')
        np.testing.assert_allclose(resampler.resample(image, coords, **kwargs), [1.0, 5.0, 5.0])
        np.testing.assert_allclose(resampler.resample(image, coords[1], **kwargs), 5.0)
        colormap = np.ones((8, 3))
        new = resampler.resample(colormap, np.full((2, 2, 1), 1.5), **kwargs)
        np.testing.assert_allclose(new, np.full((2, 2, 3), 5.0))

  def test_resample_single_point_with_boundary_antialiasing(self) -> None:
    image = np.ones((4, 4))
    for boundary in ['natural', 'constant']:
      with self.subTest(boundary=boundary):
        np.testing.assert_allclose(resampler.resample(image, [0.5, 0.6], boundary=boundary), 1.0)
        np.testing.assert_allclose(resampler.resample(image, [0.5, 1.2], boundary=boundary), 0.0)

  def test_resample_with_jacobian_and_blocks(self) -> None:
    rng = np.random.default_rng(1)
    jacobian = np.broadcast_to(np.eye(2), (30, 30, 2, 2))
    coords = rng.random((30, 30, 2))
    new = resampler.resample(rng.random((8, 8)), coords, jacobian=jacobian, max_block_size=100)
    _check_eq(new.shape, (30, 30))

  def test_resample_block_partitioning_is_exact(self) -> None:
    rng = np.random.default_rng(1)
    array = rng.random((5, 7, 3))
    coords = rng.random((30, 20, 2))
    blocked = resampler.resample(array, coords, max_block_size=50)
    unblocked = resampler.resample(array, coords, max_block_size=0)
    np.testing.assert_allclose(blocked, unblocked, rtol=0, atol=1e-12)

  def test_resize_noop_only_if_filter_and_dtype_are_unchanged(self) -> None:
    array = np.random.default_rng(1).random((5, 6))
    assert resampler.resize(array, array.shape) is array
    new = resampler.resize(array, array.shape, filter='gaussian', prefilter='lanczos3')
    assert not np.allclose(new, array)  # The (non-interpolating) filter is applied.
    new = resampler.resize(array.astype(np.float32), array.shape, dtype=np.float64)
    _check_eq(new.dtype, np.float64)

  def test_minification_with_cardinal_prefilter_applies_digital_filter(self) -> None:
    array = np.random.default_rng(1).random(40)
    new = resampler.resize(array, (13,), filter='lanczos3', prefilter='cardinal3')
    reconstructed = resampler.resize(array, (13,), filter='lanczos3', prefilter='bspline3')
    expected = resampler._apply_digital_filter_1d(
        reconstructed,
        resampler._get_gridtype('dual'),
        resampler._get_boundary('clamp'),  # The 'auto' boundary for minification.
        0.0,
        resampler._get_filter('cardinal3'),
    )
    np.testing.assert_allclose(new, expected, rtol=0, atol=1e-6)  # (The kernel is sampled.)

  def test_digital_filter_of_single_sample(self) -> None:
    for filter, boundary in [('omoms3', 'reflect'), ('cardinal3', 'clamp')]:
      with self.subTest(filter=filter, boundary=boundary):
        new = resampler.resize(np.array([0.3]), (3,), filter=filter, boundary=boundary)
        np.testing.assert_allclose(new, 0.3, rtol=0, atol=1e-6)

  def test_primal_wrap_with_cardinal_filter_is_interpolating(self) -> None:
    array = np.random.default_rng(1).random(7)
    array[-1] = array[0]  # For a primal grid with 'wrap', the last sample repeats the first one.
    for filter in ['cardinal3', 'cardinal5', 'omoms3']:
      with self.subTest(filter=filter):
        kwargs: Any = dict(gridtype='primal', boundary='wrap', filter=filter)
        new = resampler.resize(array, (13,), **kwargs)
        np.testing.assert_allclose(new[::2], array, rtol=0, atol=1e-12)

  def test_uniform_resize_uses_gridtype(self) -> None:
    new = resampler.uniform_resize(np.ones((3, 5)), (3, 3), gridtype='primal', filter='triangle')
    np.testing.assert_allclose(new[:, 0], [0.0, 1.0, 0.0])

  def test_uniform_resize_rejects_scale_and_translate(self) -> None:
    array = np.ones((5, 7))
    kwargs_list: list[dict[str, Any]] = [dict(scale=2.0), dict(translate=0.1)]
    for kwargs in kwargs_list:
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        resampler.uniform_resize(array, (4, 4), **kwargs)

  def test_resize_parameter_validation_and_invariants(self) -> None:
    array = np.random.default_rng(1).random((5, 7, 3))
    expected = resampler.resize(array, (8, 4))
    with self.assertRaises(ValueError):
      resampler.resize(array, (8, 4), dim_order=[0, 0])
    np.testing.assert_allclose(resampler.resize(array, (8, 4), dim_order=[1, 0]), expected)
    np.testing.assert_allclose(resampler.resize(array, (8, 4), num_threads=1), expected)
    with self.assertRaises(ValueError):  # Both src_gamma and dst_gamma must be specified.
      resampler.resize(np.zeros((4, 4), np.uint8), (3, 3), src_gamma='srgb')

  def test_resize_rounds_negative_integers(self) -> None:
    for value in [-3, -2, 2, 3]:
      with self.subTest(value=value):
        new = resampler.resize(np.full(3, value, np.int16), (5,))
        _check_eq(new.dtype, np.int16)
        _check_eq(new, np.full(5, value, np.int16))

  def test_resize_of_uint8_with_default_gamma(self) -> None:
    new = resampler.resize(np.full((6, 6, 3), 200, np.uint8), (8, 4))
    _check_eq(new.dtype, np.uint8)
    _check_eq(new, np.full((8, 4, 3), 200, np.uint8))

  def test_resize_in_arraylib_and_jaxjit_match_numpy(self) -> None:
    array = np.random.default_rng(1).random((5, 7, 3))
    expected = resampler.resize(array, (8, 4))
    for arraylib in resampler.ARRAYLIBS:
      with self.subTest(arraylib=arraylib):
        new = resampler.resize_in_arraylib(array, (8, 4), arraylib=arraylib)
        np.testing.assert_allclose(new, expected, rtol=0, atol=1e-12)
    if 'jax' in resampler.ARRAYLIBS:
      import jax.numpy as jnp

      new = resampler.jaxjit_resize(jnp.asarray(array), (8, 4))
      np.testing.assert_allclose(np.asarray(new), expected, rtol=0, atol=1e-12)

  def test_rotate_image_about_center(self) -> None:
    image = np.random.default_rng(1).random((6, 8, 3))
    new = resampler.rotate_image_about_center(image, 0.0)
    np.testing.assert_allclose(new, image, rtol=0, atol=1e-12)
    new = resampler.rotate_image_about_center(image, np.pi / 2, new_shape=(8, 6), filter='impulse')
    _check_eq(new.shape, (8, 6, 3))
    np.testing.assert_allclose(new, np.rot90(image, 1), rtol=0, atol=1e-12)
    matrix = resampler.rotation_about_center_in_2d((6, 8), 0.3)
    _check_eq(matrix.shape, (3, 3))
    np.testing.assert_allclose(matrix @ [0.5, 0.5, 1.0], [0.5, 0.5, 1.0])  # The center is fixed.

  def test_rotate_image_multiple_times_with_new_shape(self) -> None:
    image = np.random.default_rng(1).random((4, 6))
    kwargs: Any = dict(new_shape=(6, 6), filter='triangle')
    once = resampler.rotate_image_about_center(image, 0.2, **kwargs)
    expected = resampler.rotate_image_about_center(once, 0.2, **kwargs)
    new = resampler.rotate_image_about_center(image, 0.2, num_rotations=2, **kwargs)
    np.testing.assert_allclose(new, expected, rtol=0, atol=1e-12)

  @unittest.skipIf('torch' not in resampler.ARRAYLIBS, 'Requires torch.')
  def test_torch_parameter_and_unsupported_dtype(self) -> None:
    import torch

    new = resampler.resize(torch.nn.Parameter(torch.ones(4, 4)), (2, 2))
    np.testing.assert_allclose(new.detach().numpy(), 1.0)
    with self.assertRaises(ValueError):
      resampler.resize(torch.ones(4, 4, dtype=torch.float16), (2, 2))

  @unittest.skipIf('torch' not in resampler.ARRAYLIBS, 'Requires torch.')
  def test_resize_is_differentiable_in_torch(self) -> None:
    import torch

    tensor = torch.tensor(np.random.default_rng(1).random((5, 7, 3)), requires_grad=True)
    resampler.resize(tensor, (8, 4)).sum().backward()
    assert tensor.grad is not None
    np.testing.assert_allclose(float(tensor.grad.sum()), 8 * 4 * 3)  # The rows sum to one.

  @unittest.skipIf('torch' not in resampler.ARRAYLIBS, 'Requires torch.')
  def test_gradient_of_digital_filter_in_torch(self) -> None:
    import torch

    rng = np.random.default_rng(1)
    for config in itertools.product(resampler.GRIDTYPES, ['reflect', 'wrap', 'clamp']):
      gridtype, boundary = config
      with self.subTest(config=config):
        tensor = torch.tensor(rng.random(6), requires_grad=True)
        kwargs: Any = dict(gridtype=gridtype, boundary=boundary, filter='cardinal3')
        func: Any = functools.partial(resampler.resize, shape=(11,), **kwargs)
        assert torch.autograd.gradcheck(func, (tensor,))

  @unittest.skipIf('torch' not in resampler.ARRAYLIBS, 'Requires torch.')
  @unittest.skipIf(not resampler._USING_NUMBA, 'The fast box downsampling requires numba.')
  def test_fast_box_downsampling_matches_general_path(self) -> None:
    array = np.random.default_rng(1).random((12, 8, 3)).astype(np.float32)
    for filter in ['box', 'trapezoid']:
      with self.subTest(filter=filter):
        fast = resampler.resize(array, (3, 2), filter=filter)
        general = resampler.resize_in_arraylib(array, (3, 2), filter=filter, arraylib='torch')
        np.testing.assert_allclose(fast, general, rtol=0, atol=1e-6)
        expected = array.reshape(3, 4, 2, 4, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(fast, expected, rtol=0, atol=1e-6)


if __name__ == '__main__':
  unittest.main()
