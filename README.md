# Study of Active Dispersion

## B.Tech Capstone Project

Department: Chemical Engineering
Institute: Indian Institute of Technology Ropar
Project: Study of Active Dispersion

## Project Overview

This project focuses on the computational study of active dispersion of microorganisms using microscopy image analysis and active-particle modelling.

The objective is to extract quantitative information about microorganism motion from microscopy image sequences and investigate how active motion contributes to dispersion and transport.

## Project Objectives

- Process microscopy images of microorganisms.
- Identify and isolate the experimental region.
- Detect and segment microorganism structures from the image background.
- Extract individual object positions and trajectories.
- Determine velocity and orientation from tracked motion.
- Quantify dispersion using trajectory-based analysis.
- Study Brownian, inertial and active-particle models.
- Compare experimentally observed motion with theoretical model predictions.

## Current Computational Workflow

The image-processing workflow currently implemented consists of:

1. Petri-dish region detection using circular region-of-interest detection.
2. Circular masking to isolate the experimental region.
3. Background intensity removal.
4. Local contrast enhancement using CLAHE.
5. Gaussian filtering for image smoothing.
6. Black-hat enhancement for highlighting dark microorganism structures.
7. Otsu thresholding for binary segmentation.
8. Morphological processing and removal of small unwanted structures.
9. Generation of preliminary segmentation masks.

## Current Project Status

The image-processing stage has been implemented and tested on a representative microscopy frame.

The current workflow produces:

- Petri-dish ROI detection and masking.
- Background-reduced images.
- Contrast-enhanced images.
- Binary segmentation.
- Morphologically processed segmentation masks.
- Preliminary microorganism structures.

A current limitation is the dense central region of the image, where multiple microorganism structures can become connected during segmentation. Reliable separation of individual objects is therefore required before final trajectory-based analysis.

## Next Steps

The remaining computational workflow is:

Object detection -> Tracking -> Velocity and orientation -> Dispersion analysis -> Brownian, inertial and active modelling -> Experimental-model comparison

The next stage will focus on reliable individual object detection and tracking across consecutive microscopy frames.

## Repository Structure

02_Image_Processing

- Image preprocessing scripts
- Segmentation scripts
- ROI and masking outputs
- Background-removal outputs
- Contrast-enhancement outputs
- Binary segmentation outputs
- Morphological processing outputs

The repository currently documents the implemented image-processing stage. Additional project stages will be added as they are developed and validated.

## Data Note

The raw microscopy dataset is not included in this public repository. The repository contains the computational scripts and selected processing outputs used to document the image-processing workflow.

## Tools

- Python
- OpenCV
- NumPy
- scikit-image
- SciPy
- Active-matter computational analysis tools