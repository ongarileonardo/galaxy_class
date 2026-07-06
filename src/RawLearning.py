# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Supervised Learning
#
# This notebook use simple supervised learning to classify images into classes.

# %%
# install requirements

# if connected to colab

if 'google.colab' in str(get_ipython()):
  print('running on colab')
  # !git clone https://github.com/ongarileonardo/galaxy_class.git
  # %cd galaxy_class

# %pip install --upgrade --force-reinstall -r requirements.txt

# %%
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# import everything we need first
from keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from astroNN.models import Galaxy10CNN
from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion

# %%
# To load images and labels (will download automatically at the first time)
# First time downloading location will be ~/.astroNN/datasets/
images, labels = galaxy10.load_data()

# %%
images.shape

# %%
# divide images and labels into training and test sets

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, random_state=42)

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

idx = np.random.randint(0, y_train.shape[0], size=10)

for ax, i in zip(axes.flat, idx):
    ax.imshow(X_train[i])
    ax.set_title(f"Class {y_train[i]}\n{galaxy10cls_lookup(y_train[i])}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
# plt.savefig("original.pdf")
plt.show()


# %%
def center_crop(img, crop_size):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h + crop_size, start_w:start_w + crop_size]

crop_size = 128
X_train_cropped = np.array([center_crop(img, crop_size) for img in X_train])
X_test_cropped = np.array([center_crop(img, crop_size) for img in X_test])

print(X_train_cropped.shape)

# %%
from PIL import Image
import numpy as np

X_train_gray = np.array([
    np.array(Image.fromarray(img).convert('L')) for img in X_train_cropped
])
X_test_gray = np.array([
    np.array(Image.fromarray(img).convert('L')) for img in X_train_cropped
])

print(X_train_gray.shape)

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for ax, i in zip(axes.flat, idx):
    ax.imshow(X_train_gray[i], cmap="gray")
    ax.set_title(f"Class {y_train[i]}\n{galaxy10cls_lookup(y_train[i])}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data Cleaning
#
# 1. normalization (min-max)
# 2. normalization using percentiles
# 3. gaussian / bilateral filter
# 4. histogram equalization

# %%
# Min-max normalization (per-image)

X_min = X_train_gray.min(axis=(1,2), keepdims=True)  
X_max = X_train_gray.max(axis=(1,2), keepdims=True)
X_norm = (X_train_gray - X_min) / (X_max - X_min + 1e-8)  

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for ax, i in zip(axes.flat, idx):
    ax.imshow(X_norm[i], cmap="gray")
    ax.set_title(f"Class {y_train[i]}\n{galaxy10cls_lookup(y_train[i])}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
# plt.savefig("gray_cropped.pdf")
plt.show()

# %%
X_norm.shape

# %%
# gaussian filter on images

from scipy.ndimage import gaussian_filter

X_gauss = gaussian_filter(X_norm, sigma=1, axes=[1,2])

# %%
X_gauss.min(), X_gauss.max()

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for ax, i in zip(axes.flat, idx):
    ax.imshow(X_gauss[i], cmap="gray")
    ax.set_title(f"Class {y_train[i]}\n{galaxy10cls_lookup(y_train[i])}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
# plt.savefig("gaussian_smooth.pdf")
plt.show()

# %%
# train a model on X_norm (without filters) and X_gauss

X_norm.shape, y_train.shape

# %%
# train a CNN model

import tensorflow as tf
from tensorflow.keras import layers, models

# %%

input_shape = (X_norm.shape[1], X_norm.shape[1], 1)
model = models.Sequential()

model.add(layers.Input(shape=(256, 256, 1)))

# block 1
model.add(layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# block 2
model.add(layers.Conv2D(32, (5, 5), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

# block 3
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# converts feature maps → single vector per feature map
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))

# model.add(layers.Dropout(0.4))

model.add(layers.Dense(10, activation='softmax'))

# %%
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# %%
X_norm.shape

# %%
X_flat = X_norm[..., np.newaxis]
y_labels = utils.to_categorical(y_train, 10)


# %%
X_flat.shape

# %%
history = model.fit(
    X_flat, y_labels,
    epochs=20,
    batch_size=128,
    shuffle=True
)

# %%
y = model.predict(X_flat)

# %%
y_labels = (y > 0.5).astype(int)

# %%
y_labels = np.argmax(y_labels, axis=1)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_labels, y_train)

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.show()
