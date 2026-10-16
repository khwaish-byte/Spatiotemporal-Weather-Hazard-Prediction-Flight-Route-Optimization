# AI-Based Cloud Segmentation and Cyclone-Aware Flight Route Planning

A research prototype that combines satellite-image segmentation,
cyclone-image regression experiments, and pathfinding algorithms to
explore how weather information could support flight-route planning.

> **Important safety note:** This project is for learning,
> demonstration, and research only. It is **not an operational aviation
> or weather-warning system**. Do not use its outputs to make real
> flight, diversion, or safety decisions.

## Overview

The Python file contains several experiments combined into one large,
Google Colab-style script:

1.  **Cloud segmentation** using a U-Net model with a ResNet34 encoder.
2.  **Grid-based route planning** using the A\* search algorithm.
3.  **Risk-weighted route planning** that adds a penalty to cells
    assigned higher risk.
4.  **Cyclone intensity regression experiments** using a ResNet18-based
    neural network.
5.  **Airport suitability estimation** using a fully connected neural
    network.
6.  **Indian flight disruption scoring** using a neural network trained
    on heuristic-generated labels.
7.  **Visualization experiments** for cloud masks, routes, and simulated
    altitude/risk maps.

These sections are exploratory experiments, not a single fully
integrated production pipeline. Some risk maps and training labels are
synthetic or generated from hand-written formulas.

## How It Works

### 1. Cloud segmentation

The cloud-segmentation experiment downloads the **38-Cloud Cloud
Segmentation in Satellite Images** dataset through `kagglehub`.

For each satellite image patch, it loads four spectral bands:

-   Red
-   Green
-   Blue
-   Near-infrared (NIR)

The bands are stacked into a four-channel image and normalized. The
corresponding ground-truth mask is converted into a binary
cloud/non-cloud mask.

**Model:** U-Net with a ResNet34 encoder initialized with ImageNet
weights.

-   Input channels: 4
-   Output channels: 1
-   Output: a pixel-level cloud mask
-   Loss: a weighted combination of Binary Cross-Entropy and Dice loss
-   Optimizer: AdamW
-   Metrics: Dice score and Intersection over Union (IoU), with accuracy
    also calculated in the training code

The dataset is split into training and validation subsets. The training
code includes variants that use GPU acceleration and, in one version,
mixed-precision training.

**Scope limitation:** The 38-Cloud labels teach the model to segment
clouds. They do not directly label cyclone boundaries, cyclone tracks,
or cyclone intensity. Cloud segmentation alone is not cyclone detection.

### 2. A\* route planning

The A\* algorithm searches a grid for a path between a start cell and a
destination cell.

It combines:

-   **G-score:** cost accumulated from the start to the current cell.
-   **Heuristic:** estimated remaining distance to the destination
    (Euclidean distance in this implementation).
-   **F-score:** the sum of the G-score and heuristic.

The implementation explores neighboring grid cells, including diagonal
moves. Cells marked as blocked are avoided. The cloud mask can be
converted into a grid, and binary dilation is used in an experiment to
expand marked regions before pathfinding.

A\* is a **search/pathfinding algorithm**, not a machine-learning model.

### 3. Risk-weighted A\*

A second pathfinding experiment adds a penalty for cells assigned higher
risk. A risk weight controls how strongly the search avoids those cells.
The code also includes a simulated storm region for demonstration.

Some risk maps and storm regions are artificially generated or blurred.
They should not be interpreted as calibrated meteorological hazard
predictions.

### 4. Cyclone intensity regression

The file includes experiments using a **ResNet18-based regression
model** on infrared cyclone imagery.

The network adapts the first convolution to accept a single-channel
image and replaces the classification head with a regression head that
predicts a numerical value. The experiments use regression losses such
as Mean Squared Error (MSE) or Smooth L1 loss.

**Important data caveat:** At least one experiment creates target
wind-speed values from image brightness using a hand-written mapping.
Those labels are synthetic, not measured wind speeds. Performance on
those labels does not demonstrate accurate real-world cyclone-intensity
prediction.

### 5. Airport suitability and flight disruption

The file also contains fully connected neural networks for two scoring
experiments:

-   **Airport suitability:** features include aircraft-to-airport
    distance, airport-to-storm distance, elevation, facility-type score,
    and remaining fuel.
-   **Flight disruption:** features include origin/destination
    coordinates and an estimated route distance.

The airport suitability labels are generated using a hand-written
formula. The flight disruption labels are also derived from a heuristic
based on distance to an assumed cyclone center. These models learn the
generated rules; their outputs are not validated operational
probabilities.

### 6. Altitude and fuel-aware route experiments

Additional demonstrations compare route costs against assumed fuel-range
and altitude limits. Some cloud-altitude values are estimated from a
synthetic risk map rather than measured cloud-top heights. These
experiments illustrate possible algorithm designs, not real aircraft
performance or airspace constraints.

## Datasets Used

The script references these Kaggle datasets:

-   [38-Cloud Cloud Segmentation in Satellite
    Images](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images)
-   [INSAT-3D Infrared Raw Cyclone Images
    (2013--2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)
-   [Tropical Cyclone Intensity
    Regression](https://www.kaggle.com/datasets/kbdharun/tropical-cyclone-intensity-regression)
-   [Airports in
    India](https://www.kaggle.com/datasets/shivammittal274/airports-in-india)
-   [Indian Flight
    Schedules](https://www.kaggle.com/datasets/nikhilkhetan/indian-flight-schedules)

Dataset availability, file layout, licensing, and access requirements
can change. Review each dataset's terms before using or redistributing
it.

## Technology Stack

-   Python
-   PyTorch
-   `segmentation-models-pytorch`
-   `torchvision`
-   NumPy
-   pandas
-   scikit-learn
-   `tifffile`
-   Pillow
-   Matplotlib
-   SciPy
-   `kagglehub`
-   `tqdm`
-   `h5py` (for dataset inspection/experiments)

The exact dependencies depend on which section of the script you run.

## Environment and Setup

The source file was exported from a Google Colab notebook and contains
notebook-specific commands such as `!pip install`, `display(...)`,
Google Drive mounting, and multiple experiment sections. It may also
contain runtime-reset code. **It is not a clean, standalone Python
script that should be executed from top to bottom unchanged.**

### Recommended environment

Use Google Colab or a Jupyter notebook. A GPU is recommended for
training the deep-learning models. A CUDA-enabled GPU is optional for
some sections, but the training code may take much longer on CPU.

### Install packages

In a Colab/Jupyter cell, install the packages needed for the experiment
you want to run:

``` python
%pip install kagglehub tifffile segmentation-models-pytorch torch torchvision numpy pandas scikit-learn matplotlib scipy pillow tqdm h5py
```

The source file contains several repeated imports, installation cells,
and experiments. Before running it:

1.  Choose the specific experiment you want to run.
2.  Separate that experiment into notebook cells or a clean Python
    module.
3.  Remove notebook-only shell commands (`!pip install`) from ordinary
    `.py` code, or run them as notebook cells.
4.  Review and remove runtime-reset commands such as
    `os.kill(os.getpid(), 9)` before execution.
5.  Check hard-coded dataset paths and replace them with paths valid in
    your environment.
6.  Confirm that the expected dataset files and columns exist.
7.  Start with a small dataset subset and a short training run.

### Dataset access

The code uses `kagglehub.dataset_download(...)` to download datasets.
Depending on the dataset and your environment, Kaggle authentication or
account access may be required. The script also contains hard-coded
paths in some later experiments; these must be updated if your download
location differs.

## Running the Experiments

Because the file combines multiple experiments, there is no single
reliable command that runs every section end-to-end without
modification.

A sensible order is:

1.  Run the cloud-segmentation dataset inspection and confirm that the
    four image bands and ground-truth masks are found.
2.  Run the U-Net training and validation section.
3.  Visualize predicted cloud masks and compare them with ground-truth
    masks.
4.  Run A\* on a small, clearly defined grid to verify the pathfinding
    implementation.
5.  Add risk penalties using a documented, explicitly simulated risk
    map.
6.  Run cyclone regression only after confirming that the target values
    are genuine, correctly matched measurements.
7.  Treat airport suitability and disruption networks as demonstrations
    until their labels are replaced with reliable real-world targets and
    evaluated independently.

The exact cell or section to run depends on which experiment you want to
reproduce.

## Evaluation Metrics

-   **Accuracy:** fraction of pixels classified correctly; can be
    misleading when cloud and non-cloud pixels are imbalanced.
-   **Dice score:** measures overlap between the predicted and reference
    cloud masks.
-   **Intersection over Union (IoU):** intersection area divided by
    union area of predicted and reference masks.
-   **Regression loss:** measures numerical prediction error during
    cyclone regression experiments.
-   **Route cost/length:** can compare candidate paths in the grid, but
    is not equivalent to real flight time, fuel burn, or aviation
    safety.

Report validation results with the dataset split, preprocessing,
threshold, and evaluation method. Do not describe scores on synthetic
labels as evidence of real-world predictive accuracy.

## Current Limitations

-   The source file contains multiple versions and experiments, rather
    than one clean end-to-end application.
-   Cloud segmentation is not the same task as cyclone detection or
    cyclone forecasting.
-   Some cyclone-intensity and risk targets are generated by formulas
    rather than observed ground truth.
-   The route planner operates on image/grid cells and does not
    implement a complete geospatial flight-routing system.
-   The code does not include validated air-traffic, NOTAM, wind-field,
    turbulence, regulatory, aircraft-performance, or live weather
    integrations.
-   Simplified coordinate-distance calculations and assumed thresholds
    are not suitable for operational navigation.
-   Model performance must be tested on independent, correctly labelled
    data before making scientific claims.

## Suggested Future Improvements

1.  Refactor the script into separate modules for data loading,
    segmentation, cyclone modelling, route planning, evaluation, and
    visualization.
2.  Add a reproducible `requirements.txt` and configuration file.
3.  Use separate training, validation, and test sets, avoiding data
    leakage.
4.  Verify that cyclone targets are measured observations aligned with
    each image's timestamp and location.
5.  Evaluate cyclone detection separately from cloud segmentation, with
    appropriate labelled cyclone data.
6.  Replace synthetic route-risk labels with carefully sourced,
    validated meteorological and aviation data.
7.  Use geospatial coordinates and suitable distance calculations rather
    than treating image pixels as real-world distances.
8.  Save model checkpoints, preprocessing settings, dataset versions,
    random seeds, and evaluation reports.
9.  Compare predictions against independent baselines and document
    failure cases.
10. Keep all route outputs in a simulation-only environment until
    validated by qualified aviation and meteorological experts.

## Project Summary

This project demonstrates how computer vision, neural networks, and
graph-search algorithms can be combined in a weather-aware
flight-planning research prototype. Its strongest clearly defined ML
component is **satellite cloud segmentation with U-Net and a ResNet34
encoder**. The route planner uses **A**\* and risk-weighted search,
while the cyclone, airport, and disruption sections are experimental and
require stronger ground-truth data and validation before their
predictions can be trusted.

## Disclaimer

This repository is an educational prototype. It does not provide
certified meteorological forecasts, flight plans, airport
recommendations, or aviation safety advice. Never use it to operate or
reroute an aircraft.
