# Resampler Notebook

[Hugues Hoppe](https://hhoppe.com/)
&nbsp;&nbsp; Aug 2022.

[**[Open in Colab]**](https://colab.research.google.com/github/hhoppe/resampler/blob/main/resampler.ipynb)
&nbsp; [**[in Kaggle]**](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/resampler/blob/main/resampler.ipynb)
&nbsp; [**[in MyBinder]**](https://mybinder.org/v2/gh/hhoppe/resampler/main?filepath=resampler.ipynb)
&nbsp; [**[in DeepNote]**](https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fresampler%2Fblob%2Fmain%2Fresampler.ipynb)
&nbsp; [**[GitHub source]**](https://github.com/hhoppe/resampler)
&nbsp; [**[API docs]**](https://hhoppe.github.io/resampler/)
&nbsp; [**[PyPI package]**](https://pypi.org/project/resampler/)

This Python notebook has several roles:
- Source code for the `resampler` library.
- Illustrated documentation.
- Usage examples.
- Unit tests.
- Signal-processing experiments to justify choices.
- Lint, build, and export the package and its documentation.


# Overview of resampler library

`resampler` enables fast differentiable resizing and warping of arbitrary grids.
It supports:

- grids of arbitrary dimension (e.g., 1D audio, 2D images, 3D video, 4D batches of videos),
  containing

- sample values of arbitrary shape
  (e.g., scalars, RGB colors, motion vectors, Jacobian matrices) and

- arbitrary numeric type (integer, floating, and complex);

- either `dual` ("half-integer") or `primal` **grid-type**
  for each dimension;

- many **boundary** rules,
  specified per dimension, extensible via subclassing;

- an extensible set of parameterized **filter** kernels,
  selectable per dimension;

- optional **gamma** transfer functions
  for correct linear-space filtering;

- prefiltering for accurate antialiasing when downsampling;

- processing within several **array libraries**
  (`numpy`, `tensorflow`, and `torch`);

- efficient backpropagation of **gradients**
  for both `tensorflow` and `torch`;

- easy installation, without any native-code extension module, yet

- **faster resizing** than the C++ implementations
  in `tf.image`, `torch.nn`, and `torchvision`.


## Example usage

```python
!pip install -q mediapy resampler
import mediapy as media
import numpy as np
import resampler
```

```python
array = np.random.rand(4, 4, 3)  # 4x4 RGB image.
upsampled = resampler.resize(array, (128, 128))  # To 128x128 resolution.
media.show_images({'4x4': array, '128x128': upsampled}, height=128)
```
> <img src="https://drive.google.com/uc?export=download&id=1tXm7Z8_ILYpTOsW1a5Z4S-Dvd1vcn7Q5"/>

```python
image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
downsampled = resampler.resize(image, (32, 32))
media.show_images({'128x128': image, '32x32': downsampled}, height=128)
```
> <img src="https://drive.google.com/uc?export=download&id=1OiVNvszGZP3COh8mhI0dd2v00cMw2TA0"/>

```python
import matplotlib.pyplot as plt
```

```python
array = [3.0, 5.0, 8.0, 7.0]
new_dual = resampler.resize(array, (32,))  # (default gridtype='dual') 8x resolution.
new_primal = resampler.resize(array, (25,), gridtype='primal')  # 8x resolution.
_, axs = plt.subplots(1, 2, figsize=(7, 1.5))
axs[0].set_title('gridtype dual')
axs[0].plot((np.arange(len(array)) + 0.5) / len(array), array, 'o')
axs[0].plot((np.arange(len(new_dual)) + 0.5) / len(new_dual), new_dual, '.')
axs[1].set_title('gridtype primal')
axs[1].plot(np.arange(len(array)) / (len(array) - 1), array, 'o')
axs[1].plot(np.arange(len(new_primal)) / (len(new_primal) - 1), new_primal, '.')
```
> <img src="https://drive.google.com/uc?export=download&id=1VGjyX2nvBKaWyGbrMt3g0Nd3G1YdtFjg"/>

```python
batch_size = 4
batch_of_images = media.moving_circle((16, 16), batch_size)
spacer = np.ones((64, 16, 3))
upsampled = resampler.resize(batch_of_images, (batch_size, 64, 64))
media.show_images([*batch_of_images, spacer, *upsampled], border=True, height=64)
```
> <img src="https://drive.google.com/uc?export=download&id=1PLHu5mCpmb-_54ybvfr6kLUUTHD6l73t"/>

```python
media.show_videos({'original': batch_of_images, 'upsampled': upsampled}, fps=1)
```
> original<img src="https://drive.google.com/uc?export=download&id=1WCwwbgYZordX14-XvHiV2Gc_60I1KD39"/>&nbsp; upsampled<img src="https://drive.google.com/uc?export=download&id=11Of3Gbv6p2BTxJD2rO0zAWEEv4w3BIe5"/>

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


**Advanced usage:**

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
> <img src="https://drive.google.com/uc?export=download&id=1WUsrghao2Py9hSCPWfinVYg6Lga55h1X"/>

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
> <img src="https://drive.google.com/uc?export=download&id=1vqnNGeAw5uTNvMEt8hzQY3uXOJugMtJY"/>


**Limitations:**

- Filters are assumed to be [separable](https://en.wikipedia.org/wiki/Separable_filter).
For rotation equivariance (e.g., bandlimit the signal uniformly in all directions),
it would be nice to support the (non-separable) 2D rotationally symmetric
[sombrero function](https://en.wikipedia.org/wiki/Sombrero_function)
$f(\textbf{x}) = \text{jinc}(\|\textbf{x}\|)$,
where $\text{jinc}(r) = 2J_1(\pi r)/(\pi r)$.
(The Fourier transform of a circle
[involves the first-order Bessel function of the first kind](
  https://en.wikipedia.org/wiki/Airy_disk).)
