# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test Reampler Notebook
#
# [**[Open in Colab]**](https://colab.research.google.com/github/hhoppe/resampler/blob/main/resampler_example_usage_notebook.ipynb)
# &nbsp; [**[GitHub source]**](https://github.com/hhoppe/resampler)

# %% [markdown]
# ## Example Usage

# %% tags=[]
# # !pip install -q mediapy resampler
# !pip install -q mediapy
import mediapy as media
import numpy as np
import resampler

# %% tags=[]
rng = np.random.default_rng(seed=1)
array = rng.random((4, 6, 3))  # 4x4 RGB image.
upsampled = resampler.resize(array, (128, 192))  # To 128x192 resolution.
media.show_images({'4x6': array, '128x192': upsampled}, height=128)

# %% tags=[]
image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
downsampled = resampler.resize(image, (32, 32))
media.show_images({'128x128': image, '32x32': downsampled}, height=128)

# %% tags=[]
import matplotlib.pyplot as plt

# %% tags=[]
array = [3.0, 5.0, 8.0, 7.0]  # 4 source samples in 1D.
new_dual = resampler.resize(array, (32,))  # (default gridtype='dual') 8x resolution.
new_primal = resampler.resize(array, (25,), gridtype='primal')  # 8x resolution.
_, axs = plt.subplots(1, 2, figsize=(9, 1.5))
axs[0].set_title('gridtype dual')
axs[0].plot((np.arange(len(array)) + 0.5) / len(array), array, 'o')
axs[0].plot((np.arange(len(new_dual)) + 0.5) / len(new_dual), new_dual, '.')
axs[1].set_title('gridtype primal')
axs[1].plot(np.arange(len(array)) / (len(array) - 1), array, 'o')
axs[1].plot(np.arange(len(new_primal)) / (len(new_primal) - 1), new_primal, '.')
plt.show()

# %% tags=[]
batch_size = 4
batch_of_images = media.moving_circle((16, 16), batch_size)
spacer = np.ones((64, 16, 3))
upsampled = resampler.resize(batch_of_images, (batch_size, 64, 64))
media.show_images([*batch_of_images, spacer, *upsampled], border=True, height=64)

# %% [markdown]
# <!-- For Emacs:
# Local Variables: *
# fill-column: 100 *
# End: *
# -->
