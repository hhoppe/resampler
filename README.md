[Hugues Hoppe](https://hhoppe.com/)
&nbsp;&nbsp; Aug 2022.

[**[Open in Colab]**](https://colab.research.google.com/github/hhoppe/resampler/blob/main/resampler_notebook.ipynb)
&nbsp;
[**[Kaggle]**](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/resampler/blob/main/resampler_notebook.ipynb)
&nbsp;
[**[MyBinder]**](https://mybinder.org/v2/gh/hhoppe/resampler/main?filepath=resampler_notebook.ipynb)
&nbsp;
[**[DeepNote]**](https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fresampler%2Fblob%2Fmain%2Fresampler_notebook.ipynb)
&nbsp;
[**[GitHub source]**](https://github.com/hhoppe/resampler)
&nbsp;
[**[API docs]**](https://hhoppe.github.io/resampler/)
&nbsp;
[**[PyPI package]**](https://pypi.org/project/resampler/)

The notebook
[<samp>resampler_notebook.ipynb</samp>](https://colab.research.google.com/github/hhoppe/resampler/blob/main/resampler_notebook.ipynb)
hosts the source code for the
[<samp>resampler</samp> library](https://pypi.org/project/resampler/),
interleaved with docs, usage examples, unit tests, and experiments.

# Overview

The `resampler` library enables fast differentiable resizing and warping of arbitrary grids.
It supports:

- grids of **any dimension** (e.g., 1D, 2D images, 3D video, 4D batches of videos), containing

- **samples of any shape** (e.g., scalars, colors, motion vectors, Jacobian matrices) and

- any **numeric type** (integer, floating, and complex);

- either `'dual'` ("half-integer") or `'primal'` **grid-type** for each dimension;

- many **boundary** rules, specified per dimension, extensible via subclassing;

- an extensible set of **filter** kernels, selectable per dimension;

- optional **gamma** transfer functions for correct linear-space filtering;

- prefiltering for accurate **antialiasing** when `resize` downsampling;

- processing within several **array libraries**
  (`numpy`, `tensorflow`, `torch`, and `jax`);

- efficient backpropagation of **gradients**
  for `tensorflow`, `torch`, and `jax`;

- few dependencies (only `scipy`) and **no native code**, yet

- **faster resizing** than C++ implementations
  in `tf.image`, `torch.nn`, and `torchvision`.

A key strategy is to leverage existing sparse matrix representations and operations.

## Example usage

```python
!pip install -q mediapy resampler
import mediapy as media
import numpy as np
import resampler
```

```python
array = np.random.default_rng(1).random((4, 6, 3))  # 4x6 RGB image.
upsampled = resampler.resize(array, (128, 192))  # To 128x192 resolution.
media.show_images({'4x6': array, '128x192': upsampled}, height=128)
```
> <img src="https://github.com/hhoppe/resampler/raw/main/media/example_array_upsampled.png"/>

```python
image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
downsampled = resampler.resize(image, (32, 32))
media.show_images({'128x128': image, '32x32': downsampled}, height=128)
```
> <img src="https://github.com/hhoppe/resampler/raw/main/media/example_array_downsampled.png"/>

```python
import matplotlib.pyplot as plt
```

```python
array = [3.0, 5.0, 8.0, 7.0]  # 4 source samples in 1D.
new_dual = resampler.resize(array, (32,))  # (default gridtype='dual') 8x resolution.
new_primal = resampler.resize(array, (25,), gridtype='primal')  # 8x resolution.
_, axs = plt.subplots(1, 2, figsize=(7, 1.5))
axs[0].set_title('gridtype dual')
axs[0].plot((np.arange(len(array)) + 0.5) / len(array), array, 'o')
axs[0].plot((np.arange(len(new_dual)) + 0.5) / len(new_dual), new_dual, '.')
axs[1].set_title('gridtype primal')
axs[1].plot(np.arange(len(array)) / (len(array) - 1), array, 'o')
axs[1].plot(np.arange(len(new_primal)) / (len(new_primal) - 1), new_primal, '.')
plt.show()
```
> <img src="https://github.com/hhoppe/resampler/raw/main/media/examples_1d_upsampling.png"/>

```python
batch_size = 4
batch_of_images = media.moving_circle((16, 16), batch_size)
upsampled = resampler.resize(batch_of_images, (batch_size, 64, 64))
media.show_videos({'original': batch_of_images, 'upsampled': upsampled}, fps=1)
```
> original
  <img src="https://github.com/hhoppe/resampler/raw/main/media/batch_original.gif"/>
  upsampled
  <img src="https://github.com/hhoppe/resampler/raw/main/media/batch_upsampled.gif"/>

Most examples above use the default
`resize()` settings:
- `gridtype='dual'` for both source and destination arrays,
- `boundary='auto'`
  which uses `'reflect'` for upsampling and `'clamp'` for downsampling,
- `filter='lanczos3'`
  (a [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling) kernel with radius 3),
- `gamma=None` which by default uses the `'power2'`
  transfer function for the `uint8` image in the second example,
- `scale=1.0, translate=0.0` (no domain transformation),
- default `precision` and output `dtype`.


## Advanced usage:

Map an image to a wider grid using custom `scale` and `translate` vectors,
with horizontal `'reflect'` and vertical `'natural'` boundary rules,
providing a constant value for the exterior,
using different filters (Lanczos and O-MOMS) in the two dimensions,
disabling gamma correction, performing computations in double-precision,
and returning an output array in single-precision:

```python
new = resampler.resize(
    image, (128, 512), boundary=('natural', 'reflect'), cval=(0.2, 0.7, 0.3),
    filter=('lanczos3', 'omoms5'), gamma='identity', scale=(0.8, 0.25),
    translate=(0.1, 0.35), precision='float64', dtype='float32')
media.show_images({'image': image, 'new': new})
```
> <img src="https://github.com/hhoppe/resampler/raw/main/media/example_advanced_usage1.png"/>

Warp an image by transforming it using
[polar coordinates](https://en.wikipedia.org/wiki/Polar_coordinate_system):

```python
shape = image.shape[:2]
yx = ((np.indices(shape).T + 0.5) / shape - 0.5).T  # [-0.5, 0.5]^2
radius, angle = np.linalg.norm(yx, axis=0), np.arctan2(*yx)
angle += (0.8 - radius).clip(0, 1) * 2.0 - 0.6
coords = np.dstack((np.sin(angle) * radius, np.cos(angle) * radius)) + 0.5
resampled = resampler.resample(image, coords, boundary='constant')
media.show_images({'image': image, 'resampled': resampled})
```
> <img src="https://github.com/hhoppe/resampler/raw/main/media/example_warp.png"/>

## Limitations:

- Filters are assumed to be [separable](https://en.wikipedia.org/wiki/Separable_filter).
- Although `resize` implements prefiltering, `resample` does not yet have it (and therefore
  may have aliased results if downsampling).
