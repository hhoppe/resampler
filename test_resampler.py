#!/usr/bin/env python3
# -*- fill-column: 100; -*-
"""Tests for package resampler.

c:/windows/sysnative/wsl -e bash -lc 'flake8 --indent-size 2 --max-line-length=1000 --extend-ignore E302,E741,E131,E305,E402 test_resampler.py && python3 test_resampler.py'
"""
from collections.abc import Callable
import functools
import itertools
import math
import os
from typing import Any
import unittest
import warnings

import numpy as np
import numpy.typing
import scipy

import resampler

_ArrayLike = numpy.typing.ArrayLike
_NDArray = numpy.typing.NDArray[Any]
_TensorflowTensor = Any

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

  def test_resize_on_list(self) -> None:
    lst = [3.0, 5.0, 8.0, 7.0]
    expected = np.array([2.84536097, 3.6902174, 5.58573019, 7.77282572, 7.8097826, 6.79608312])
    np.testing.assert_allclose(resampler.resize(lst, (6,)), expected)
    np.testing.assert_allclose(resampler.resize(np.array(lst), (6,)), expected)

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
    rng = np.random.default_rng(0)
    array = rng.random(shape, np.float32) * 2 * radius - radius
    result = {'expected': func(array), 'scipy': scipy_interp(array), 'obtained': func2(array)}

    assert all(a.dtype == np.float32 for a in result.values())
    assert all(a.shape == shape for a in result.values())
    assert np.allclose(result['scipy'], result['expected'], rtol=0, atol=1e-6)
    assert np.allclose(result['obtained'], result['expected'], rtol=0, atol=1e-6)

  def test_downsample_in_2d_using_box_filter(self) -> None:
    for shape in [(6, 6), (4, 4)]:
      for ch in [1, 2, 3, 4]:
        array = np.ones((*shape, ch), np.float32)
        new = resampler._downsample_in_2d_using_box_filter(array, (2, 2))
        _check_eq(new.shape, (2, 2, ch))
        assert np.allclose(new, 1.0)

    for shape in [(6, 6), (4, 4)]:
      array = np.ones(shape, np.float32)
      new = resampler._downsample_in_2d_using_box_filter(array, (2, 2))
      _check_eq(new.shape, (2, 2))
      assert np.allclose(new, 1.0)

  def test_block_shape_with_min_size(self) -> None:
    for compact in [True, False]:
      with self.subTest(compact=compact):
        shape = 2, 3, 4
        for min_size in range(1, math.prod(shape) + 1):
          block_shape = resampler._block_shape_with_min_size(shape, min_size, compact=compact)
          assert np.all(np.array(block_shape) >= 1)
          assert np.all(block_shape <= shape)
          assert min_size <= math.prod(block_shape) <= math.prod(shape)

  def test_split_2d(self) -> None:
    numpy_array = np.random.default_rng(0).choice([1, 2, 3, 4], (5, 8))
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
    numpy_array = np.random.default_rng(0).choice([1, 2, 3, 4], shape)

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
              src_size,
              dst_size,
              src_gridtype=resampler.DualGridtype(),
              dst_gridtype=resampler.DualGridtype(),
              boundary=resampler._get_boundary('reflect'),
              filter=resampler._get_filter('lanczos3'),
              translate=0.8,
              dtype=np.float32,
              arraylib=arraylib,
          )[0]

        results = {}
        for arraylib in resampler.ARRAYLIBS:
          sparse_matrix = resize_matrix(arraylib)
          match arraylib:
            case 'numpy':
              result = sparse_matrix.toarray()
            case 'tensorflow':
              import tensorflow as tf

              result = tf.sparse.to_dense(sparse_matrix).numpy()
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
        if cval_weight is None:
          row_sum = np.asarray(resize_matrix.sum(axis=1)).ravel()
          assert np.allclose(row_sum, 1.0, rtol=0, atol=1e-6), (resize_matrix.todense(), row_sum)

  def test_linear_precision_of_1d_primal_upsampling(self) -> None:
    array = np.arange(7.0)
    new = resampler.resize(array, (13,), gridtype='primal', filter='triangle')
    with np.printoptions(linewidth=300):
      _check_eq(new, np.arange(13) / 2)

  def test_linear_precision_of_2d_primal_upsampling(self) -> None:
    shape = 3, 5
    new_shape = 5, 9
    array = np.moveaxis(np.indices(shape, np.float32), 0, -1) @ [10, 1]
    new = resampler.resize(array, new_shape, gridtype='primal', filter='triangle')
    with np.printoptions(linewidth=300):
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
    array = np.random.default_rng(0).random(shape)
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
          coords = np.full(shape, 0.4)
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
    array = np.random.default_rng(0).random(shape)
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
    configs.append((resampler.pil_image_resize, 'lanczos3'))
    configs.append((resampler.cv_resize, 'lanczos4'))
    configs.append((resampler.scipy_ndimage_resize, 'cardinal3'))
    configs.append((resampler.skimage_transform_resize, 'cardinal3'))
    configs.append((resampler.tf_image_resize, 'lanczos3'))
    configs.append((resampler.torch_nn_resize, 'sharpcubic'))
    configs.append((resampler.jax_image_resize, 'lanczos3'))
    for config in configs:
      resizer, filter = config
      if resizer not in resampler._RESIZERS.values():  # Skip if the package is not installed.
        continue
      with self.subTest(config=config):
        tol: Any = dict(rtol=0, atol=1e-7)
        np.allclose(resizer(np.ones((11,)), (13,), filter=filter), np.ones((13,)), **tol)
        np.allclose(resizer(np.ones((8, 8)), (5, 20), filter=filter), np.ones((5, 20)), **tol)
        np.allclose(resizer(np.ones((9, 8, 3)), (13, 7), filter=filter), np.ones((13, 7, 3)), **tol)


if __name__ == '__main__':
  unittest.main()
