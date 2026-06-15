# Multidimensional Marine Heatwave Detector (Xarray + Dask) (xrMHW)

A high-performance, parallelized Python implementation of the Marine Heatwave (MHW) detection definitions proposed by [Hobday et al. (2016)](https://doi.org/10.1016/j.pocean.2015.12.014). Code developed by Gutiérrez-Cárdenas, GS. (ORCID: 0000-0002-3915-7684; DEC 2025)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-GPL3.0-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🌊 Overview

This tool is designed to process **large-scale oceanographic datasets** (global ocean circulation models, eddy-resolving models, SST data) that are multidimensional (Latitude, Longitude, Depth, Time). Unlike standard 1D implementations, this package leverages `xarray` and `dask` to:

1.  **Vectorize** detection logic across spatial grids.
2.  **Parallelize** computations to handle memory-intensive files (e.g., GLORYS, OISST).
3.  **Detect** MHW or MCS events in 3D (Surface) or 4D (Volumetric) arrays.

## 🚀 Features

- **Standard Compliance:** Implements the Hobday et al. (2016) baseline.
- **Volumetric Analysis:** Capable of detecting subsurface heatwaves if depth data is provided.
- **Robustness:** Automatically handles leap years, missing data (NaNs), and land masks.
- **Efficiency:** Uses `xr.apply_ufunc` for optimized compiled execution.

## 🛠 Installation

Install the dependencies:

```bash
pip install xarray dask netCDF4 scipy pandas numpy
```

⚙️ Input Specifications
The script expects a NetCDF file with the following characteristics:

| Dimension | Description                                                                          |
| :-------- | :----------------------------------------------------------------------------------- |
| **Time**  | Must be a datetime object. The script handles daily resolution.                      |
| **Space** | Standard `latitude` and `longitude`. Optional `depth` dimension for volumetric data. |

| Variable        | Description                                          | Units   |
| :-------------- | :--------------------------------------------------- | :------ |
| **Temperature** | Sea Surface Temperature (SST)/Potential Temperature. | °C or K |

## 🖥 Usage

To integrate the **xrMHW** detector into your oceanographic analysis workflow, follow these steps:

1. Clone the repository
   Download the source code directly from GitHub:

```bash
!conda install git -y
!git clone https://github.com/Gabo2000s/xarray_MHW-xrMHW-.git
```

2. Open xrMHW.py.

Edit or copy the CONFIGURATION section at the bottom of the script:

```bash
# --- CONFIGURATION ---
INPUT_FILE = './data/GLORYS_1993-2024.nc'
OUTPUT_FILE = './results/MHW_output.nc'
VAR_NAME = 'thetao'  # Variable name in your NetCDF
CLIM_START = 1993
CLIM_END = 2022
```

## 📊 Output Specifications

The output is a NetCDF file containing the computed MHW metrics. The data is masked: grid points without an active MHW event are filled with NaN.

| Variable                 | Name in NetCDF      | Description                                             | Units    |
| :----------------------- | :------------------ | :------------------------------------------------------ | :------- |
| **Intensity**            | `mhw_intensity`     | Temperature anomaly above climatology during the event. | °C       |
| **Max Intensity**        | `mhw_max_intensity` | The highest anomaly recorded during the event duration. | °C       |
| **Cumulative Intensity** | `mhw_cum_intensity` | Integrated intensity over the duration of the event.    | °C days  |
| **Duration**             | `mhw_duration`      | Length of the MHW event.                                | Days     |
| **Category**             | `mhw_category`      | Severity category (1=Moderate, 2=Strong, 3=Severe...).  | Unitless |

Also include the climatology and threshold

## 🔬 Methodology

### Scientific Definition

The algorithm implements the standard hierarchical definition by **Hobday et al. (2016)**:

- **Climatology:** Calculated over a user-defined baseline (default 1991-2020) using an 11-day centered window and a 31-day moving average smoothing.
- **Threshold:** temperatures exceeding the 90th percentile for MHW and 10th percentile for MCS
- **Gap handling:** the algorinthm performs a pre-filter joining, physically bridges gaps of up to 2 days (default) before checking the minimum duration of the events. The original Hobaday el al. (2016) implementation joins events after filtering. That ensures that long, significant events interrupted by a biref drop in temperature are correcly identified as a single continuous event, rather than being discrded as two invalid short fragments.
- **Event definition:** After the gap filling, any continuous period satisfying the threshold conditio at lesat 5 days is recorded as an event.

### Technical Implementation

- **High-Performance Parallelism:** Using xarray with `dask = parallelized`, the tool maps the detection logic across the time dimension, allowing the workloead to be distribuited across all available CPU cores or a cluster, handling N-dimensional arrays (latitude, longitude, time, depth) efficiently
- **Fast exit optimization:** The wrapper function includes a "Land mask cehcking". If the time series contain only `NaN`, the algorithm retunr `NaN` array, savin massive computation time during the process of big datasets
- **Wrapper:** `xarray.apply_ufunc` maps this 1D function across the **Time** dimension of the N-dimensional input array, effectively looping over `Lat`, `Lon`, and `Depth` in C-speed (vectorized).
- **Chunking:** The workflow relies on `Dask` chunking to process datasets significantly larger than the available RAM by streaming blocks of spatial data.

## 📄 License

Distributed under the GPL-3.0 license. See LICENSE for more information.

## 📚 References

Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography, 141, 227-238. DOI: 10.1016/j.pocean.2015.12.014

Schlegel, R. W., & Smit, A. J. (2018). marineHeatWaves: Detecting and analyzing marine heatwaves. Journal of Open Source Software, 3(27), 812.
