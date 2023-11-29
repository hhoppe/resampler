# %% [markdown]
# # resampler: Example usage
#
# [**[Open in Colab]**](https://colab.research.google.com/github/hhoppe/resampler/blob/main/example_usage.ipynb)
# &nbsp;
# [**[Kaggle]**](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/resampler/blob/main/example_usage.ipynb)
# &nbsp;
# [**[MyBinder]**](https://mybinder.org/v2/gh/hhoppe/resampler/main?filepath=example_usage.ipynb)
# &nbsp;
# [**[DeepNote]**](https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fresampler%2Fblob%2Fmain%2Fexample_usage.ipynb)
# &nbsp;
# [**[GitHub source]**](https://github.com/hhoppe/resampler)
# &nbsp;
# [**[API docs]**](https://hhoppe.github.io/resampler/)
# &nbsp;
# [**[PyPI package]**](https://pypi.org/project/resampler/)

# %%
# !pip install -q mediapy resampler

# %%
"""Simple examples of `resampler` usage."""

import collections
import math

import mediapy as media
import numpy as np
import matplotlib.pyplot as plt

import resampler

# %% [markdown]
# ### Upsample (magnify) an image

# %%
array = np.random.default_rng(1).random((4, 6, 3))  # 4x6 RGB image.
upsampled = resampler.resize(array, (128, 192))  # To 128x192 resolution.
media.show_images({'original 4x6': array, 'upsampled 128x192': upsampled}, height=128)

# %% [markdown]
# ### Downsample (minify) an image

# %%
image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
downsampled = resampler.resize(image, (32, 32))
media.show_images({'original 128x128': image, 'downsampled 32x32': downsampled}, height=128)

# %%
yx = (np.moveaxis(np.indices((96, 192)), 0, -1) + (0.5, 0.5)) / 96
radius = np.linalg.norm(yx - (0.75, 0.5), axis=-1)
array = np.cos((radius + 0.1) ** 0.5 * 70.0) * 0.5 + 0.5
downsampled = resampler.resize(array, (24, 48))
media.show_images(
    {'original 96x192': array, 'downsampled 24x48': downsampled}, height=96, vmin=0, vmax=1
)

# %% [markdown]
# ### Upsample a 1D array

# %%
a = [3.0, 5.0, 8.0, 7.0]  # 4 source samples in 1D.
new_dual = resampler.resize(a, (32,))  # (default gridtype='dual') 8x resolution.
new_primal = resampler.resize(a, (25,), gridtype='primal')  # 8x resolution.

# %%
_, axs = plt.subplots(1, 2, figsize=(7, 1.5))
axs[0].set(title="gridtype='dual'")
axs[0].plot((np.arange(len(a)) + 0.5) / len(a), a, 'o')
axs[0].plot((np.arange(len(new_dual)) + 0.5) / len(new_dual), new_dual, '.')
axs[1].set(title="gridtype='primal'")
axs[1].plot(np.arange(len(a)) / (len(a) - 1), a, 'o')
axs[1].plot(np.arange(len(new_primal)) / (len(new_primal) - 1), new_primal, '.')

# %% [markdown]
# ### Upsample a video

# %%
batch_size = 4
batch_of_images = media.moving_circle((16, 16), batch_size)
upsampled = resampler.resize(batch_of_images, (batch_size, 64, 64))
media.show_videos({'original': batch_of_images, 'upsampled': upsampled}, fps=1)

# %% [markdown]
# ### Advanced resizing options

# %% [markdown]
# Map an image to a wider grid using custom `scale` and `translate` vectors,
# with horizontal `'reflect'` and vertical `'natural'` boundary rules,
# providing a constant value for the exterior,
# using different filters (Lanczos and O-MOMS) in the two dimensions,
# disabling gamma correction, performing computations in double-precision,
# and returning an output array in single-precision:

# %%
new = resampler.resize(
    image,
    (128, 512),
    boundary=('natural', 'reflect'),
    cval=(0.2, 0.7, 0.3),
    filter=('lanczos3', 'omoms5'),
    gamma='identity',
    scale=(0.8, 0.25),
    translate=(0.1, 0.35),
    precision='float64',
    dtype='float32',
)
media.show_images({'original image': image, 'result of resize()': new})

# %% [markdown]
# ### Translate an image

# %%
videos = collections.defaultdict(list)
for filter in ['impulse', 'triangle', 'lanczos3']:
  for angle in np.linspace(0.0, math.tau, 20, endpoint=False):
    translate = np.array([np.cos(angle), np.sin(angle)]) * (0.6 / image.shape[0])
    videos[f"filter='{filter}'"].append(
        resampler.resize(image, image.shape[:2], filter=filter, translate=translate)
    )

media.show_videos(videos, fps=20, height=image.shape[0] * 2)

# %% [markdown]
# ### Translate with upscaling

# %%
videos = collections.defaultdict(list)
for filter in ['impulse', 'triangle', 'trapezoid', 'lanczos3']:
  for angle in np.linspace(0.0, math.tau, 20, endpoint=False):
    translate = np.array([np.cos(angle), np.sin(angle)]) * (0.6 / image.shape[0]) + [-0.4, 0.0]
    scale = 2.0
    videos[f"filter='{filter}'"].append(
        resampler.resize(image, image.shape[:2], filter=filter, translate=translate, scale=scale)
    )

media.show_videos(videos, fps=20, height=image.shape[0] * 2)

# %% [markdown]
# ### Translate with downscaling

# %%
videos = collections.defaultdict(list)
for filter in ['impulse', 'triangle', 'trapezoid', 'lanczos3']:
  for angle in np.linspace(0.0, math.tau, 20, endpoint=False):
    translate = np.array([np.cos(angle), np.sin(angle)]) * (0.6 / image.shape[0]) + [0.0, 0.0]
    new_shape = np.array(image.shape[:2]) // 2
    videos[f"filter='{filter}'"].append(
        resampler.resize(image, new_shape, filter=filter, translate=translate)
    )

media.show_videos(videos, fps=20, height=image.shape[0] * 2)

# %% [markdown]
# ### Rotate an image

# %%
videos = collections.defaultdict(list)
for filter in ['impulse', 'triangle', 'lanczos3']:
  for angle in np.sin(np.linspace(0.0, math.tau, 60, endpoint=False)) * math.radians(6):
    videos[f"filter='{filter}'"].append(
        resampler.rotate_image_about_center(image, angle, filter=filter, boundary='constant')
    )

media.show_videos(videos, fps=20, height=image.shape[0] * 2)

# %% [markdown]
# ### Warp with general resampling

# %% [markdown]
# Warp an image by transforming it using
# [polar coordinates](https://en.wikipedia.org/wiki/Polar_coordinate_system):

# %%
shape = image.shape[:2]
yx = ((np.indices(shape).T + 0.5) / shape - 0.5).T  # [-0.5, 0.5]^2
radius, angle = np.linalg.norm(yx, axis=0), np.arctan2(*yx)
angle += (0.8 - radius).clip(0, 1) * 2.0 - 0.6
coords = np.dstack((np.sin(angle) * radius, np.cos(angle) * radius)) + 0.5
resampled = resampler.resample(image, coords, boundary='constant')
media.show_images({'original image': image, 'resampled': resampled}, height=image.shape[0] * 2)
