# Study of Active Dispersion

## B.Tech Capstone Project

- **Department:** Chemical Engineering
- **Institute:** Indian Institute of Technology Ropar
- **Project:** Study of Active Dispersion
- **Supervisor:** Dr Vishwajeet Mehendia

## Project Overview

This repository contains microscopy image-processing work and computational active-particle modelling carried out for the project.

The implemented image-processing code performs preprocessing and preliminary segmentation of microscopy images. The modelling code implements overdamped and inertial active Brownian particle simulations and includes numerical analysis of simulation outputs.

## Repository Structure

| Folder | Contents |
|---|---|
| `Experimental_Data` | Supporting experimental files and a representative microscopy image |
| `Image_Processing` | Image preprocessing and segmentation scripts, along with selected processing outputs |
| `Modelling` | Active-particle simulation code, numerical methods, analysis scripts, tests, documentation, and selected results |

## Image Processing

The image-processing workflow includes:

1. Petri-dish region detection and masking.
2. Background intensity removal.
3. Local contrast enhancement using CLAHE.
4. Gaussian filtering.
5. Black-hat enhancement.
6. Otsu thresholding.
7. Morphological processing.
8. Generation of preliminary segmentation masks.

The segmentation workflow has been tested on representative microscopy images. Dense central regions can cause multiple structures to become connected, limiting reliable separation of individual objects.

The scripts expect TIFF input images in the image-processing directory. The complete raw microscopy sequence is not included in this repository.

## Computational Modelling

The `Modelling` directory contains implementations and analyses of active-particle systems, including:

- Overdamped and inertial active Brownian particle simulations.
- Numerical methods and simulation utilities.
- Simulation parameter-sensitivity studies.
- Mean-squared displacement analysis.
- Time-averaged mean-squared displacement analysis.
- Orientation and velocity autocorrelation analysis.
- Comparisons with theoretical behaviour.
- Numerical tests and selected simulation results.

The modelling documentation describes the implemented methods, simulation configurations, outputs, assumptions, and limitations.

## Experimental Data

The complete raw microscopy dataset is not included because of its size.

The original dataset contains 1,266 TIFF images: one background image and 1,265 numbered experimental frames. Its approximate total size is 7.39 GB.

The repository includes three supporting files and a representative PNG frame in `Experimental_Data/raw_data`.

The correspondence between the supporting files and the complete image sequence, as well as the spatial calibration and frame timing, has not been established sufficiently for quantitative experimental validation.

## Data and Modelling Limitations

- The image-processing outputs document preprocessing and preliminary segmentation.
- Dense image regions can cause multiple objects to merge in segmentation masks.
- Large simulation trajectory files are not all included in the repository.
- The documented seed-42 simulation analysis uses a single random seed and should not be interpreted as a multi-seed uncertainty analysis.
- Simulation distances are expressed in model length units; physical length calibration is not provided.
- Simulation assumptions and numerical limitations are described in the documentation under `Modelling`.

## Tools and Libraries

- Python
- NumPy
- SciPy
- OpenCV
- scikit-image
- Matplotlib
- pytest
