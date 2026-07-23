# Galaxy Morphology and Rotational Symmetry

## Project Overview

This project uses the original astroNN Galaxy10 SDSS dataset of galaxy images and labels to build a supervised learning model for galaxy morphology classification.

The goal is to train a model that can predict galaxy classes from images, compare results with and without data augmentation, and explore improvements over the Galaxy10CNN baseline from astroNN.

## Usage

Dependencies for Galaxy10Learning and Visualization can be found in the `galaxy10-requirements.txt` file; for other notebooks, use `user-requirements.txt`. It is recommended to use 2 different virtual environments based on the requirements, to prevent dependencies version mismatch.

### Training
For network training, it is strictly recommended to use a Colab environment or a Jupyter Lab environment with a GPU access.

**Recommended**: use anaconda or miniconda to create a virtual environment

```bash
       > conda create -n myenv python=3.11
       > conda activate myenv
(myenv)> pip install -r requirements-learn.txt
(myenv)> jupyter lab
```

The default user-requirements.txt file will install tensorflow 2.15 with CUDA support only for Linux. 

If you are using a different OS, you will need to check your system requirements and install the appropriate version of tensorflow and CUDA/GPU support, otherwise the training will use the CPU only.

To monitor the GPU activity, run the following command 
`watch -n 1 nvidia-smi` 

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

## Notes
The dataset used in the project is the original Galaxy10 SDSS file containing 21,785 RGB images of size 69×69 pixels.

This can be imported manually downloading the file from http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5, or loading it with the library command `astroNN.datasets.load_galaxy10sdss()`

Two different requirements files are needed to solve the issue with astroNN tensorflow version dependency, that generates a conflict with other packages needed.


## References

[1] Dieleman, Willett, and Dambre, "Rotation-invariant convolutional neural networks for galaxy morphology prediction"

[2] LEUNG, Henry W.; BOVY, Jo. Deep learning of multi-element abundances from high-resolution spectroscopic data. Monthly Notices of the Royal Astronomical Society, 2019, 483.3: 3255-3277.