# Multidimensional Marine Heatwave Detector (Xarray + Dask) (xmhw)

A high-performance, parallelized Python implementation of the Marine Heatwave (MHW) detection definitions proposed by [Hobday et al. (2016)](https://doi.org/10.1016/j.pocean.2015.12.014).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🌊 Overview

This tool is designed to process **large-scale oceanographic datasets** (global models, satellite data) that are multidimensional (Latitude, Longitude, Depth, Time). Unlike standard 1D implementations, this package leverages `xarray` and `dask` to:

1.  **Vectorize** detection logic across spatial grids.
2.  **Parallelize** computations to handle memory-intensive files (e.g., GLORYS, OISST).
3.  **Detect** events in 3D (Surface) or 4D (Volumetric) arrays.

## 🚀 Features

* **Standard Compliance:** Implements the Hobday et al. (2016) baseline.
* **Volumetric Analysis:** Capable of detecting subsurface heatwaves if depth data is provided.
* **Robustness:** Automatically handles leap years, missing data (NaNs), and land masks.
* **Efficiency:** Uses `xr.apply_ufunc` for optimized compiled execution.

## 🛠 Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com//marine-heatwave-detector.git](https://github.com/tu-usuario/marine-heatwave-detector.git)
cd marine-heatwave-detector
pip install -r requirements.txt
```

⚙️ Input Specifications
The script expects a NetCDF file with the following characteristics:

| Dimension | Description |
| :--- | :--- |
| **Time** | Must be a datetime object. The script handles daily resolution. |
| **Space** | Standard `latitude` and `longitude`. Optional `depth` dimension for volumetric data. |

| Variable | Description | Units |
| :--- | :--- | :--- |
| **Temperature** | Sea Surface Temperature (SST)/Potential Temperature. | °C or K |

## 🖥 Usage
Open xmhw.py.

Edit the CONFIGURATION section at the bottom of the script:

```bash
# --- CONFIGURATION ---
INPUT_FILE = './data/GLORYS_1993-2024.nc'
OUTPUT_FILE = './results/MHW_output.nc'
VAR_NAME = 'thetao'  # Variable name in your NetCDF
CLIM_START = 1993
CLIM_END = 2022
```

Run the script:
```bash
python xmhw.py
```

📊 Output Specifications
The output is a NetCDF file containing the computed MHW metrics. The data is masked: grid points without an active MHW event are filled with NaN.

| Variable | Name in NetCDF | Description | Units |
| :--- | :--- | :--- | :--- |
| **Intensity** | `mhw_intensity` | Temperature anomaly above climatology during the event. | °C |
| **Max Intensity** | `mhw_max_intensity` | The highest anomaly recorded during the event duration. | °C |
| **Cumulative Intensity** | `mhw_cum_intensity` | Integrated intensity over the duration of the event. | °C days |
| **Duration** | `mhw_duration` | Length of the MHW event. | Days |
| **Category** | `mhw_category` | Severity category (1=Moderate, 2=Strong, 3=Severe...). | Unitless |

## 🔬 Methodology

### Scientific Definition
The algorithm follows the hierarchical definition by **Hobday et al. (2016)**:

* **Climatology:** Calculated over a user-defined baseline (default 1993-2022) using an 11-day centered window.
* **Threshold:** The 90th percentile of the baseline temperatures.
* **Event:** Periods where temperature exceeds the threshold for at least 5 consecutive days.
* **Gaps:** Events broken by gaps of less than 3 days are merged.

### Technical Implementation
* **Chunking:** The data is chunked using `Dask` to allow processing of datasets larger than RAM.
* **Core Logic:** The detection function (`detect_mhw_core`) is a pure `NumPy` implementation optimized for 1D arrays.
* **Wrapper:** `xarray.apply_ufunc` maps this 1D function across the **Time** dimension of the N-dimensional input array, effectively looping over `Lat`, `Lon`, and `Depth` in C-speed (vectorized).

## 📄 License
Distributed under the MIT License. See LICENSE for more information.

## 📚 References
Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography, 141, 227-238. DOI: 10.1016/j.pocean.2015.12.014

Schlegel, R. W., & Smit, A. J. (2018). marineHeatWaves: Detecting and analyzing marine heatwaves. Journal of Open Source Software, 3(27), 812.
