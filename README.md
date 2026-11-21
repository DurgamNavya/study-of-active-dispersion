# Study of Active Dispersion

## B.Tech Capstone Project

- **Department:** Chemical Engineering
- **Institute:** Indian Institute of Technology Ropar
- **Project:** Study of Active Dispersion

## Project Overview

This project investigates the motion and dispersion of microorganisms through microscopy image processing and computational active-particle modelling.

The computational work includes image preprocessing and segmentation, supporting experimental tracking data, and simulations of Brownian, active Brownian, and inertial active-particle systems.

## Project Objectives

- Process microscopy images and identify the experimental region.
- Detect and segment microorganism structures.
- Develop methods for extracting object positions and trajectories.
- Study velocity, orientation, and dispersion.
- Implement and analyse active-particle models.
- Compare simulation behaviour with theoretical predictions and, where validated data are available, experimental observations.

## Image Processing

The implemented image-processing workflow includes:

1. Petri-dish region detection and masking.
2. Background intensity removal.
3. Local contrast enhancement using CLAHE.
4. Gaussian filtering.
5. Black-hat enhancement.
6. Otsu thresholding.
7. Morphological processing.
8. Generation of preliminary segmentation masks.

The current segmentation workflow has been tested on representative microscopy images. Dense central regions can cause multiple structures to become connected, so reliable individual-object separation remains a limitation.

## Computational Modelling

The `06_Modelling` directory contains:

- Overdamped and inertial active Brownian particle simulation code.
- Numerical methods and simulation utilities.
- Parameter-sensitivity studies.
- Mean-squared displacement and correlation analysis.
- Theoretical comparisons and validation scripts.
- Tests for the numerical core, simulation runner, and analysis pipeline.

The modelling documentation describes the simulation assumptions, parameters, outputs, and limitations.

## Repository Structure

| Folder | Contents |
|---|---|
| `01_Experimental_Data` | Supporting tracking files and a representative image; full raw image sequence is stored separately |
| `02_Image_Processing` | Preprocessing, segmentation scripts, and selected image outputs |
| `03_Tracking` | Reserved for validated tracking workflows and outputs |
| `04_Velocity_Orientation` | Reserved for velocity and orientation analysis |
| `05_Dispersion_Analysis` | Reserved for experimental dispersion analysis |
| `06_Modelling` | Active-particle simulations, numerical methods, analysis scripts, tests, and selected results |
| `07_AMEP` | Reserved for project-specific AMEP analysis files |
| `08_Report_PPT` | Reserved for finalized project reports and presentation material |

## Experimental Data Availability

The complete raw microscopy dataset is not included in this repository because of its size.

The original dataset contains 1,266 TIFF images, including one background image and 1,265 numbered experimental frames. Its approximate total size is 7.39 GB.

Three smaller supporting files and a representative PNG frame are included in `01_Experimental_Data/raw_data`.

The correspondence between supporting tracking files and the complete image sequence, along with spatial calibration and frame timing, must be established before using them for quantitative experimental validation.

## Data and Results Limitations

- Selected image outputs are provided to document the processing workflow.
- Some large simulation trajectory files are excluded from the repository because of GitHub file-size limits.
- Experimental tracking, velocity, and dispersion results should be treated as quantitative results only after the relevant calibration and data provenance have been verified.
- Simulation assumptions and limitations are documented in `06_Modelling`.

## Tools and Libraries

- Python
- NumPy
- SciPy
- OpenCV
- scikit-image
- Matplotlib
- pytest
- Active Matter Environment Package (AMEP)
