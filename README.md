# NASA C-MAPSS: Remaining Useful Life (RUL) Prediction

```{=html}
<p align="center">
```
`<strong>`{=html}Aircraft Engine Predictive Maintenance using Machine
Learning and Deep Learning`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange" alt="Scikit-learn">`{=html}
`<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">`{=html}
`<img src="https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter" alt="Jupyter">`{=html}
`<img src="https://img.shields.io/badge/Status-Work%20in%20Progress-yellow" alt="Work in progress">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Project Overview

This project explores **Remaining Useful Life (RUL) prediction** for
aircraft engines using the NASA C-MAPSS turbofan engine degradation
dataset.

The goal is to estimate how many operational cycles an engine has left
before failure, using its historical operating conditions and sensor
measurements.

The repository currently contains two notebook versions and an early
modular Python implementation:

-   **`NASAA_v1.ipynb`** -- an earlier experimental notebook. Its
    generated training and validation data have more overlap, which can
    make the evaluation less reliable and contribute to weaker
    generalization.
-   **`NASAA_v2_final.ipynb`** -- the improved notebook and the
    recommended starting point for current experiments. It uses a
    stricter engine-level validation split and a more complete workflow.
    It is still a work in progress and can be improved further.
-   **`Nasa_proj/`** -- the initial modular-code prototype. It contains
    the first attempt to separate the project into Python modules, but
    currently implements only a small part of the overall logic. The
    next major task is to rebuild and extend it using the workflow in
    v2.

> **Recommended starting point: `NASAA_v2_final.ipynb`.** Use v1 to
> understand the earlier approach and compare how the validation
> strategy affects results.

## Repository Structure

``` text
ML_project/
│
├── extracted/
│   └── CMaps/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│       └── ...                 # Other subsets, if downloaded
│
├── Nasa_proj/
│   ├── outputs/
│   ├── __init__.py
│   ├── dataset.py
│   ├── eda.py
│   ├── main.py
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── ...
│
├── NASAA_v1.ipynb
├── NASAA_v2_final.ipynb
├── archive.zip
└── README.md
```

The structure above reflects the current repository layout. The contents
of `extracted/` depend on which C-MAPSS files you have downloaded and
extracted. The `archive.zip` file is an archive; it is not required to
run the notebooks if the dataset is already extracted.

## Notebook Versions

### v1 --- Earlier Experiment

**File:** `NASAA_v1.ipynb`

-   Contains the earlier version of the project workflow.
-   Its training and validation data generation has more overlap than
    v2.
-   Results from this version should be treated as exploratory rather
    than as the final benchmark.
-   Kept in the repository to preserve the development history and allow
    comparison.

### v2 --- Recommended Experiment

**File:** `NASAA_v2_final.ipynb`

-   Provides the cleaner and more complete notebook workflow.
-   Uses `GroupShuffleSplit` with engine IDs (`unit`) to keep engines
    separated between training and validation.
-   Builds validation examples using observations available up to a
    selected cycle.
-   Explores engineered sensor features and multiple model families.
-   Includes traditional ML models (Random Forest and, when installed,
    XGBoost), a PyTorch MLP, and a PyTorch GRU.
-   Uses validation results for model selection and reserves the
    official test labels for final evaluation.

v2 is the current working version, **not a finished or permanently
optimal solution**. Feature engineering, model architecture,
hyperparameters, and validation robustness can be improved in future
experiments.

### `Nasa_proj/` --- Initial Modular Prototype

The `Nasa_proj/` directory is the first attempt to organize the project
into reusable Python modules. It currently contains files for dataset
handling, EDA, preprocessing, modeling, and execution.

At this stage, it is mainly a structural starting point and contains
limited project logic. It should not yet be considered equivalent to the
v2 notebook.

**Planned direction:** upgrade the modular implementation by
transferring and refactoring the verified workflow from v2. Keep
responsibilities separated across modules rather than copying the entire
notebook into one Python file.

## Dataset

The project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion
System Simulation) dataset.

Each observation contains:

-   `unit`: engine identifier
-   `cycle`: operational cycle
-   `op_setting_1` to `op_setting_3`: operational settings
-   `sensor_1` to `sensor_21`: sensor measurements

The notebook is currently configured for the **FD001** subset. The
dataset directory should contain the matching training, test, and RUL
files:

``` text
CMaps/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

Other subsets use corresponding filenames, such as `train_FD002.txt`,
`test_FD002.txt`, and `RUL_FD002.txt`.

The dataset is not assumed to be included in this repository. Download
it from an authorized NASA source and extract the required files
locally.

## Workflow in v2

The current v2 notebook follows this high-level process:

1.  Load the selected C-MAPSS subset.
2.  Inspect the data and perform exploratory analysis.
3.  Create training RUL labels from each engine's final recorded cycle.
4.  Engineer sensor features, including rolling statistics, differences,
    lag features, and EWMA.
5.  Split training engines into training and validation groups.
6.  Build validation examples using only the engine history available up
    to each selected cutoff.
7.  Train and compare traditional ML and PyTorch models.
8.  Select a model using validation performance.
9.  Retrain the selected model using all available training engines and
    evaluate it on the official test engines.

### Models explored

  -----------------------------------------------------------------------
  Model                   Type                    Role
  ----------------------- ----------------------- -----------------------
  Random Forest Regressor Machine Learning        Tree-based regression
                                                  baseline

  XGBoost Regressor       Machine Learning        Gradient-boosted tree
                                                  model (optional
                                                  dependency)

  MLP                     Deep Learning           Regression using
                                                  tabular features

  GRU                     Deep Learning           Regression using
                                                  sequential sensor
                                                  observations
  -----------------------------------------------------------------------

### Validation note

The v2 notebook uses an engine-level split so that an engine is assigned
to either training or validation, rather than having its rows randomly
distributed across both sets. Its validation examples are cut off before
the engine's final training cycle to simulate prediction from incomplete
operating history.

This is a more appropriate setup for evaluating generalization to unseen
engines than a row-level split. It is still one validation split;
repeated group-based validation can be added later for a more robust
estimate.

## Setup

### 1. Clone or open the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd ML_project
```

If the repository is already on your computer, open the `ML_project`
folder in VS Code.

### 2. Create and activate an environment

Using Conda:

``` bash
conda create -n cmapss python=3.12 -y
conda activate cmapss
```

Or using `venv`:

``` bash
python -m venv .venv
```

On Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install notebook dependencies

The v2 notebook uses NumPy, pandas, Matplotlib, Seaborn, scikit-learn,
and PyTorch. XGBoost is optional; without it, the notebook can still run
the Random Forest experiment.

Install the dependencies:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter ipykernel
```

Optional:

``` bash
pip install xgboost
```

For GPU training, install a PyTorch build compatible with your installed
NVIDIA driver and the CUDA version supported by the selected PyTorch
release. Check the official PyTorch installation instructions rather
than assuming that the system's reported CUDA version is the required
wheel version.

### 4. Install the modular project's dependencies (when using it)

The modular prototype has its own dependency file:

``` bash
pip install -r Nasa_proj/requirements.txt
```

Run this from the repository root. If the modular implementation is
later consolidated around one root-level environment, dependency
management can be simplified.

## Dataset Path Configuration

Before running either notebook, open the notebook and locate the dataset
configuration cell.

The current v2 notebook uses:

``` python
subset = "FD001"
data_dir = Path(r"D:\ML_project\extracted\CMaps")
```

Update `data_dir` to the actual location of your extracted `CMaps`
directory.

For example, if the repository is located at `D:\Projects\ML_project`:

``` python
data_dir = Path(r"D:\Projects\ML_project\extracted\CMaps")
```

To run another subset, change the `subset` value:

``` python
subset = "FD002"  # or FD001, FD003, FD004
```

Make sure all three matching files exist in `data_dir`:

``` text
train_<SUBSET>.txt
test_<SUBSET>.txt
RUL_<SUBSET>.txt
```

For example, when `subset = "FD002"`, the notebook expects
`train_FD002.txt`, `test_FD002.txt`, and `RUL_FD002.txt`.

**Important:** Changing the notebook's filename (for example,
`NASAA_v1.ipynb` to another name) does not select a dataset. Change the
`subset` and `data_dir` variables inside the notebook to select the data
that should be loaded.

## How to Run

### Option A --- Run v2 (recommended)

1.  Open `NASAA_v2_final.ipynb` in VS Code or Jupyter.
2.  Select the Python environment in which the dependencies are
    installed.
3.  Update `data_dir` and `subset` in the dataset configuration cell.
4.  Confirm that the three expected dataset files exist.
5.  Run the notebook from the first cell to the last cell.
6.  Review the validation comparison and final test evaluation.

### Option B --- Run v1 (historical comparison)

1.  Open `NASAA_v1.ipynb`.
2.  Select the correct Python environment.
3.  Check and update its dataset path and subset configuration.
4.  Run the notebook cells in order.
5.  Treat its results as an earlier experiment, not as the primary
    benchmark, because its data overlap makes the evaluation less
    reliable.

Keep the two notebooks as separate versions. You do not need to rename
one notebook to run the other; simply open and execute the version you
want to use. If a notebook contains a hard-coded dataset path, update
that path in that notebook.

### Option C --- Run the modular prototype

The modular code is an early prototype and may not yet reproduce all
functionality in v2.

1.  Open the repository root in VS Code.
2.  Install dependencies from `Nasa_proj/requirements.txt`.
3.  Inspect the configuration and dataset paths used by `Nasa_proj`.
4.  Run the entry point from the repository root:

``` bash
python -m Nasa_proj.main
```

If the package or entry-point setup has not been completed, inspect
`Nasa_proj/main.py` and run the currently supported entry point as
documented in that file. The modular code should be upgraded before
expecting it to match the notebook's complete results.

## Evaluation

The v2 notebook compares models using regression metrics such as:

-   **MAE (Mean Absolute Error):** average absolute difference between
    predicted and actual RUL.
-   **RMSE (Root Mean Squared Error):** gives larger errors more weight.
-   **NASA Score:** an asymmetric score that penalizes late RUL
    predictions more heavily than early predictions.

Use validation results for model selection. Keep the official test set
for the final evaluation rather than repeatedly tuning against its
labels.

No fixed performance numbers are listed here because results depend on
the selected subset, features, model configuration, and experiment run.
Add measured results to this README only after reproducing them.

## Development Roadmap

-   [x] Load and inspect the C-MAPSS data in the notebook workflow.
-   [x] Create training RUL targets.
-   [x] Explore sensor feature engineering.
-   [x] Add engine-level train/validation separation in v2.
-   [x] Experiment with traditional ML and PyTorch models in v2.
-   [ ] Improve model performance through controlled feature and
    hyperparameter experiments.
-   [ ] Repeat validation across multiple engine-level splits.
-   [ ] Upgrade `Nasa_proj/` to reproduce the v2 preprocessing and
    validation workflow.
-   [ ] Separate configuration, data preparation, feature engineering,
    training, evaluation, and inference into reusable modules.
-   [ ] Add reproducible experiment outputs and model artifacts.

## Next Development Priority

The immediate priority is to **upgrade `Nasa_proj/` using
`NASAA_v2_final.ipynb` as the reference implementation**.

A sensible order is:

1.  Reproduce v2 dataset loading and RUL target creation.
2.  Implement the same preprocessing and feature engineering.
3.  Reproduce the engine-level train/validation split and cutoff-based
    validation.
4.  Move model definitions and training loops into reusable modules.
5.  Centralize evaluation metrics and model comparison.
6.  Add a consistent inference path for the official test data.
7.  Verify that the modular pipeline produces results consistent with
    the notebook under the same configuration.

The notebook should remain useful for experimentation, while the modular
project becomes the maintainable implementation.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<strong>`{=html}Built as a practical exploration of predictive
maintenance, regression, and sequence modelling.`</strong>`{=html}
```{=html}
</p>
```
