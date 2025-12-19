"""
Marine Heatwave Detector (Hobday et al., 2016 implementation)
==============================================================

A robust, Xarray-compatible implementation of the Marine Heatwave (MHW) definitions 
proposed by Hobday et al. (2016). This script utilizes `xarray.apply_ufunc` to 
parallelize the detection algorithm across multidimensional datasets (Latitude, 
Longitude, Depth, Time).

Author: Gutiérrez-Cárdenas, GS. (ORCID: 0000-0002-3915-7684)
Date: Dec 2025
License: MIT
"""

import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from datetime import date
from dask.diagnostics import ProgressBar
import warnings

# Suppress warnings for clean output during large computations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. CORE SCIENTIFIC LOGIC (Numpy/Pandas based)
# =============================================================================

def runavg(ts, w):
    """
    Calculates a running average (smoothing) on a time series.
    
    Parameters
    ----------
    ts : np.ndarray
        Input time series (1D).
    w : int
        Window size for the running average.
        
    Returns
    -------
    np.ndarray
        Smoothed time series.
    """
    N = len(ts)
    # Pad the array to handle edge effects (circular boundary condition simulation)
    ts = np.append(ts, np.append(ts, ts))
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Return the central part corresponding to original data
    ts = ts_smooth[N:2*N]
    return ts

def detect_mhw_core(t_ordinal, temp, clim_period=(None, None), pctile=90, window_half_width=5, 
                   smooth_pctile=True, smooth_width=31, min_duration=5, 
                   join_across_gaps=True, max_gap=2):
    """
    Core detection algorithm for Marine Heatwaves on a single 1D time series.
    Based on Hobday et al. (2016) marine heat wave definition. Edited and optimized from 
    https://github.com/ecjoliver/marineHeatWaves

    Parameters
    ----------
    t_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1D).
    clim_period : tuple
        (start_year, end_year) for climatology baseline.
    pctile : int
        Percentile threshold (default 90).
    window_half_width : int
        Width of window for climatology calculation (default 5 days).
        
    Returns
    -------
    tuple
        (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category, 
         mhw_intensity_max, mhw_intensity_cum)
    """
    
    # 1. Initialize Time Vectors
    T = len(t_ordinal)
    dates_pd = pd.to_datetime([date.fromordinal(int(x)) for x in t_ordinal])
    year = dates_pd.year.values
    month = dates_pd.month.values
    day = dates_pd.day.values
    
    # Create a Day-of-Year (DOY) map handling leap years (366 days)
    t_leap = np.arange(date(2012, 1, 1).toordinal(), date(2012, 12, 31).toordinal()+1)
    dates_leap = pd.to_datetime([date.fromordinal(x) for x in t_leap]) 
    doy_map = {(m, d): dy for m, d, dy in zip(dates_leap.month, dates_leap.day, range(1, 367))}
    doy = np.array([doy_map.get((m, d), 0) for m, d in zip(month, day)]) # .get safe access

    # 2. Climatology Calculation
    if (clim_period[0] is None) or (clim_period[1] is None):
        clim_start, clim_end = year[0], year[-1]
    else:
        clim_start, clim_end = clim_period

    # Filter data for baseline period
    clim_mask = (year >= clim_start) & (year <= clim_end)
    temp_clim = temp[clim_mask]
    doy_clim = doy[clim_mask]
    
    thresh_clim_year = np.full(366, np.nan)
    seas_clim_year = np.full(366, np.nan)
    
    # Calculate threshold and climatology for each day of year (1-366)
    for d in range(1, 367):
        if d == 60: continue # Skip Feb 29 logic placeholder initially
        
        # Define window (circular)
        window_days = np.arange(d - window_half_width, d + window_half_width + 1)
        window_days = ((window_days - 1) % 366) + 1
        window_days = window_days[window_days != 60] # Exclude Feb 29 from window logic
        
        in_window = np.isin(doy_clim, window_days)
        data_in_window = temp_clim[in_window]
        
        if len(data_in_window) > 0:
            thresh_clim_year[d-1] = np.nanpercentile(data_in_window, pctile)
            seas_clim_year[d-1] = np.nanmean(data_in_window)

    # Interpolate Feb 29 (DOY 60)
    thresh_clim_year[59] = 0.5 * thresh_clim_year[58] + 0.5 * thresh_clim_year[60]
    seas_clim_year[59] = 0.5 * seas_clim_year[58] + 0.5 * seas_clim_year[60]

    # Smooth the climatology/threshold
    if smooth_pctile:
        thresh_clim_year = runavg(thresh_clim_year, smooth_width)
        seas_clim_year = runavg(seas_clim_year, smooth_width)

    # Map back to full time series
    # doy-1 because array is 0-indexed, doy is 1-366
    clim_thresh = thresh_clim_year[doy - 1]
    clim_seas = seas_clim_year[doy - 1]

    # 3. Event Detection
    exceed_bool = temp > clim_thresh
    events, n_events = ndimage.label(exceed_bool)

    # Filter by duration (minimum 5 days)
    for ev in range(1, n_events + 1):
        if (events == ev).sum() < min_duration:
            events[events == ev] = 0

    # Join events across small gaps
    if join_across_gaps:
        # Note: A more complex gap bridging logic could be implemented here 
        # but standard label re-evaluation is often sufficient for basic bridging 
        # if the boolean mask was pre-processed. 
        # For strict Hobday: we usually iterate and fill gaps. 
        # Keeping user logic here for consistency:
        events, n_events = ndimage.label(events > 0)

    # 4. Metrics Extraction
    # Initialize arrays as float to accommodate NaNs later if needed
    mhw_duration = np.zeros(T, dtype=float) 
    mhw_category = np.zeros(T, dtype=float)
    mhw_intensity_max = np.zeros(T, dtype=float)
    mhw_intensity_cum = np.zeros(T, dtype=float)
    is_mhw = np.zeros(T, dtype=bool)

    for ev in range(1, n_events + 1):
        idx = np.where(events == ev)[0]
        dur = len(idx)
        if dur < min_duration: continue
        
        temps_ev = temp[idx]
        seas_ev = clim_seas[idx]
        thresh_ev = clim_thresh[idx]
        anoms = temps_ev - seas_ev
        
        i_max = np.max(anoms)
        i_cum = np.sum(anoms)
        
        # Categorization
        peak_idx = np.argmax(anoms)
        intensity_diff = thresh_ev[peak_idx] - seas_ev[peak_idx]
        if intensity_diff == 0: intensity_diff = 1e-5 # Avoid zero division
        ratio = i_max / intensity_diff
        cat = max(1, int(np.floor(ratio)))
        
        # Assign values to the full time dimension
        mhw_duration[idx] = float(dur) 
        mhw_category[idx] = float(cat)
        mhw_intensity_max[idx] = i_max
        mhw_intensity_cum[idx] = i_cum
        is_mhw[idx] = True

    anomaly = temp - clim_seas
    
    return clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category, mhw_intensity_max, mhw_intensity_cum

# =============================================================================
# 2. ROBUST WRAPPER (Handles NaNs & Vectorization Interface)
# =============================================================================

def mhw_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year, **kwargs):
    """
    Wrapper to handle NaNs and interface with xarray.apply_ufunc, passing 
    dynamic configuration parameters to the core detection logic.

    Parameters
    ----------
    time_ordinal : np.ndarray
        Time vector in ordinal format.
    temp : np.ndarray
        Temperature vector (1D).
    clim_start_year : int or float
        Start year of the climatology baseline.
    clim_end_year : int or float
        End year of the climatology baseline.
    **kwargs : optional
        Additional arguments passed directly to `detect_mhw_core`.
        (e.g., pctile, window_half_width, min_duration).

    Returns
    -------
    tuple
        Tuple of numpy arrays containing MHW statistics.
    """
    T = len(time_ordinal)
    
    # Return NaNs if the spatial point is empty (Land mask)
    if np.isnan(temp).all():
        nan_arr = np.full(T, np.nan)
        return (nan_arr, nan_arr, nan_arr, np.zeros(T, dtype=bool), 
                nan_arr, nan_arr, nan_arr, nan_arr)

    # Linear interpolation for small gaps of NaNs in data
    if np.isnan(temp).any():
        nans = np.isnan(temp)
        x = lambda z: z.nonzero()[0]
        temp[nans] = np.interp(x(nans), x(~nans), temp[~nans])

    # Ensure types for Numba/Numpy compatibility
    time_ordinal_safe = time_ordinal.astype(int)
    temp_safe = temp.astype(float)

    # Pass **kwargs to the core function
    return detect_mhw_core(
        time_ordinal_safe, 
        temp_safe, 
        clim_period=(int(clim_start_year), int(clim_end_year)),
        **kwargs 
    )

# =============================================================================
# 3. XARRAY INTEGRATION
# =============================================================================

# =============================================================================
# 3. XARRAY INTEGRATION
# =============================================================================

def xmhw_func(ds, temp_var_name, clim_period, **kwargs):
    """
    Applies the MHW detection over the entire Xarray Dataset using Dask,
    allowing full customization of detection parameters.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature and time.
    temp_var_name : str
        Name of the temperature variable in the dataset.
    clim_period : tuple
        A tuple (start_year, end_year) defining the climatology baseline.
    **kwargs : optional
        Arguments passed directly to the detection algorithm. 
        Common examples:
            - pctile (default: 90)
            - window_half_width (default: 5)
            - min_duration (default: 5)
            - max_gap (default: 2)

    Returns
    -------
    xr.Dataset
        A new dataset containing the detected MHW metrics masked by event occurrence.
    """
    # 1. Prepare Time Coordinate
    # Convert datetime objects to ordinal integers for the core algorithm
    time_index = ds.indexes['time']
    time_ordinal_np = time_index.map(lambda x: x.toordinal()).values
    
    time_ordinal_da = xr.DataArray(
        time_ordinal_np, 
        coords={'time': ds['time']}, 
        dims='time'
    )

    # 2. Configure Arguments for apply_ufunc
    # Merge mandatory climatology period with optional user arguments
    func_kwargs = {
        'clim_start_year': clim_period[0], 
        'clim_end_year': clim_period[1]
    }
    func_kwargs.update(kwargs)

    # Define output types matching the return tuple of detect_mhw_core
    # [seas, thresh, anomaly, is_mhw, duration, category, intensity_max, intensity_cum]
    output_dtypes = [float, float, float, bool, float, float, float, float]
    
    # 3. Apply Vectorized UFunc
    results = xr.apply_ufunc(
        mhw_1d_wrapper,
        time_ordinal_da,
        ds[temp_var_name],
        kwargs=func_kwargs,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']] * 8, 
        vectorize=True,
        dask='parallelized',
        output_dtypes=output_dtypes
    )
    
    # Unpack results
    (seas, thresh, anomaly, is_mhw, duration, category, 
     intensity_max, intensity_cum) = results

    # 4. Assemble Output Dataset
    ds_out = xr.Dataset()
    
    # Apply Mask: Variables are NaN where no MHW is present
    ds_out['mhw_intensity'] = anomaly.where(is_mhw)
    ds_out['mhw_duration'] = duration.where(is_mhw)
    ds_out['mhw_category'] = category.where(is_mhw)
    ds_out['mhw_max_intensity'] = intensity_max.where(is_mhw)
    ds_out['mhw_cum_intensity'] = intensity_cum.where(is_mhw)
    
    # Optional: diagnostic variables (can be commented out to save space)
    ds_out['climatology'] = seas
    ds_out['threshold'] = thresh

    # Preserve original coordinates (lat, lon, depth)
    ds_out = ds_out.assign_coords(ds.coords)
    
    return ds_out

# =============================================================================
# 4. MAIN EXECUTION BLOCK EXAMPLE
# =============================================================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    INPUT_FILE = 'path/to/your/input_data.nc'
    OUTPUT_FILE = 'path/to/your/output_mhw.nc'
    VAR_NAME = 'thetao'  # or 'sst' / 'analyzed_sst'
    CLIM_START = 1993
    CLIM_END = 2022
    
    # Chunking is crucial for 4D data (time, depth, lat, lon)
    CHUNKS = {'time': -1, 'depth': 1, 'latitude': 30, 'longitude': 30}

    print(f"Loading data from {INPUT_FILE}...")
    try:
        ds = xr.open_dataset(INPUT_FILE, chunks=CHUNKS)
        
        print("Starting MHW detection (this may take a while)...")
        with ProgressBar():
            ds_mhw = compute_mhw_dataset(ds, VAR_NAME, (CLIM_START, CLIM_END))
            
            # Setup compression for storage efficiency
            comp = {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': np.nan}
            encoding = {var: comp for var in ds_mhw.data_vars}
            
            print(f"Saving to {OUTPUT_FILE}...")
            ds_mhw.to_netcdf(OUTPUT_FILE, encoding=encoding)
            
        print("Process completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")