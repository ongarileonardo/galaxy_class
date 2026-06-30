# Galaxy Morphology and Rotational Symmetry

## Project Overview

This project uses the original astroNN Galaxy10 SDSS dataset of galaxy images and labels to build a supervised learning model for galaxy morphology classification.

The goal is to train a model that can predict galaxy classes from images, compare results with and without data augmentation, and explore improvements over the Galaxy10CNN baseline from astroNN.

## Usage

package versions [...]

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