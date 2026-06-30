# Galaxy Morphology and Rotational Symmetry

## Project Overview

This project uses the original astroNN Galaxy10 SDSS dataset of galaxy images and labels to build a supervised learning model for galaxy morphology classification.

The goal is to train a model that can predict galaxy classes from images, compare results with and without data augmentation, and explore improvements over the Galaxy10CNN baseline from astroNN.

## Usage

Dependencies can be found in the `requirements.txt` file. To install, run `pip install -r requirements.txt` with a Python environment between 3.10 and 3.13.

It is strictly recommended to use a Colab environment or a Jupyter Lab environment with a GPU.

## Key Tasks

- Load and preprocess the Galaxy10 SDSS dataset
- Build a convolutional neural network model for classification
- Apply data augmentation techniques such as random rotations and flips
- Compare model performance with and without augmentation
- Evaluate whether the model can improve on the Galaxy10CNN benchmark

## Performance Tips

- Convert images to grayscale to reduce input size
- Consider cropping the image center to remove irrelevant background
- Use a GPU if available to speed up training

## References

[1] Dieleman, Willett, and Dambre, "Rotation-invariant convolutional neural networks for galaxy morphology prediction"

[2] LEUNG, Henry W.; BOVY, Jo. Deep learning of multi-element abundances from high-resolution spectroscopic data. Monthly Notices of the Royal Astronomical Society, 2019, 483.3: 3255-3277.