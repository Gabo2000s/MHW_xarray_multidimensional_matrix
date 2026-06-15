"""
Marine Heatwave Detector (Hobday et al., 2016 implementation)
==============================================================

A robust, Xarray-compatible implementation of the Marine Heatwave (MHW) definitions 
proposed by Hobday et al. (2016). This script utilizes `xarray.apply_ufunc` to 
parallelize the detection algorithm across multidimensional datasets (Latitude, 
Longitude, Depth, Time).

Author: Gutiérrez-Cárdenas, GS. (ORCID: 0000-0002-3915-7684)
Date: Dec 2025
Update: Feb 2026
License: GPL3.0
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

def _filter_min_duration(events, n_events, min_duration):
    """Set to 0 every labeled event shorter than `min_duration` (vectorized)."""
    if n_events == 0:
        return events
    counts = np.bincount(events.ravel())
    too_short = np.nonzero(counts < min_duration)[0]
    too_short = too_short[too_short != 0]  # never drop the background label
    if too_short.size:
        events = np.where(np.isin(events, too_short), 0, events)
    return events


def _bridge_gaps(mask, max_gap):
    """Fill interior False runs of length <= `max_gap` so the flanking True
    runs become a single continuous event. Leading/trailing gaps are never
    filled (there is nothing to join them to)."""
    mask = mask.copy()
    gaps, n_gaps = ndimage.label(~mask)
    if n_gaps:
        for sl in ndimage.find_objects(gaps):
            start, stop = sl[0].start, sl[0].stop
            if (stop - start) <= max_gap and start > 0 and stop < len(mask):
                mask[start:stop] = True
    return mask


def detect_mhw_core(t_ordinal, temp, clim_period=(None, None), pctile=90, window_half_width=5, 
                    smooth_pctile=True, smooth_width=31, min_duration=5, 
                    join_across_gaps=True, max_gap=2, cold_spells=False,
                    join_method='post-filter'):
    
    """
    Core detection algorithm for Marine Heatwaves (or Cold Spells) on a single 1D time series.
    Based on Hobday et al. (2016) marine heatwave definition. Edited and optimized from 
    https://github.com/ecjoliver/marineHeatWaves
    
    This version reproduces the reference marineHeatWaves implementation
    (Oliver, https://github.com/ecjoliver/marineHeatWaves) bit-for-bit in its
    default configuration. Gap handling is selectable via `join_method`:
      - 'post-filter' (DEFAULT, = Hobday/Oliver): apply the minimum-duration
        filter first, then join surviving events separated by <= `max_gap` days.
      - 'pre-filter' (xrMHW option): join short gaps BEFORE the duration filter,
        so a fragmented long event survives as one. This is a deliberate,
        documented departure from Hobday/Oliver, not the default.

    Parameters
    ----------
    t_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1D).
    clim_period : tuple
        (start_year, end_year) for climatology baseline.
    pctile : int
        Percentile threshold (default 90 for MHW, typically 10 for MCS).
    window_half_width : int
        Width of window for climatology calculation (default 5 days).
    smooth_pctile : bool
        Whether to smooth the climatology and threshold (default True).
    smooth_width : int
        Width of moving average window for smoothing (default 31 days).
    min_duration : int
        Minimum duration for acceptance of detected MHWs (default 5 days).
    join_across_gaps : bool
        Whether to join events separated by a gap of `max_gap` days or less (default True).
    max_gap : int
        Maximum gap length (in days) allowed to join MHWs (default 2 days).
    join_method : {'post-filter', 'pre-filter'}
        Order of gap-joining relative to the duration filter (default
        'post-filter', which matches Hobday/Oliver). See summary above.
    cold_spells : bool, optional
        If True, detects cold spells (temp < threshold). Default False.
        
    Returns
    -------
    tuple
        (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category, 
         mhw_intensity_max, mhw_intensity_cum)
    """
    
    # 1. Initialize Time Vectors
    T = len(t_ordinal)
    # Using pandas for vectorized date conversion (performance optimization)
    dates_pd = pd.to_datetime([date.fromordinal(int(x)) for x in t_ordinal])
    year = dates_pd.year.values
    month = dates_pd.month.values
    day = dates_pd.day.values
    
    # Create a Day-of-Year (DOY) map handling leap years (366 days)
    # This aligns non-leap years to the 366-day grid (e.g., Mar 1 is always index 61)
    t_leap = np.arange(date(2012, 1, 1).toordinal(), date(2012, 12, 31).toordinal()+1)
    dates_leap = pd.to_datetime([date.fromordinal(x) for x in t_leap]) 
    doy_map = {(m, d): dy for m, d, dy in zip(dates_leap.month, dates_leap.day, range(1, 367))}
    doy = np.array([doy_map.get((m, d), 0) for m, d in zip(month, day)])

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
    
    # Calculate threshold and climatology for each day of year (1-366).
    # Windowing matches Oliver/Hobday exactly: for each occurrence of day-of-year
    # `d` in the baseline, pool the +/- window_half_width CONSECUTIVE calendar days
    # around it (index-space window), across all baseline years. This naturally
    # includes Feb-29 data in neighbouring windows, unlike a DOY-value window.
    clim_idx = np.flatnonzero(clim_mask)
    cs, ce = clim_idx[0], clim_idx[-1]
    offsets = np.arange(-window_half_width, window_half_width + 1)
    for d in range(1, 367):
        if d == 60:
            continue  # Feb 29 handled by interpolation below
        occ = np.flatnonzero(doy[cs:ce + 1] == d)
        if occ.size == 0:  # day-of-year absent (e.g. 360-day calendars)
            continue
        tt = (cs + occ[:, None] + offsets[None, :]).ravel()
        tt = tt[(tt >= 0) & (tt < T)]
        data_in_window = temp[tt]
        thresh_clim_year[d - 1] = np.nanpercentile(data_in_window, pctile)
        seas_clim_year[d - 1] = np.nanmean(data_in_window)

    # Interpolate Feb 29 (DOY 60)
    thresh_clim_year[59] = 0.5 * thresh_clim_year[58] + 0.5 * thresh_clim_year[60]
    seas_clim_year[59] = 0.5 * seas_clim_year[58] + 0.5 * seas_clim_year[60]

    # Smooth the climatology/threshold 
    if smooth_pctile:
            # Moving average function with periodic extension
            def simple_runavg(ts, w):
                N = len(ts)
                ts_triple = np.concatenate((ts, ts, ts))
                ts_smooth = np.convolve(ts_triple, np.ones(w)/w, mode='same')
                return ts_smooth[N:2*N]
                
            # Aplicación del suavizado a los umbrales y climatología estacional
            thresh_clim_year = simple_runavg(thresh_clim_year, smooth_width)
            seas_clim_year = simple_runavg(seas_clim_year, smooth_width)

    # Map back to full time series
    clim_thresh = thresh_clim_year[doy - 1]
    clim_seas = seas_clim_year[doy - 1]

    # 3. Event Detection (MHW or Cold Spell)
    if cold_spells:
        exceed_bool = temp < clim_thresh
    else:
        exceed_bool = temp > clim_thresh
    # NaN climatology days (e.g. absent DOYs) must never count as exceedances.
    exceed_bool = np.asarray(exceed_bool) & ~np.isnan(clim_thresh)

    if join_across_gaps and join_method == 'pre-filter':
        # xrMHW option: bridge short gaps BEFORE the duration filter, so an
        # otherwise-fragmented long event survives as one. This is a deliberate
        # departure from Hobday/Oliver (documented), not the default.
        exceed_bool = _bridge_gaps(exceed_bool, max_gap)
        events, n_events = ndimage.label(exceed_bool)
        events = _filter_min_duration(events, n_events, min_duration)
    else:
        # Hobday/Oliver default ('post-filter'): apply the duration filter FIRST,
        # then join the surviving events across gaps <= max_gap. Joined events are
        # >= min_duration by construction, so no second duration filter is needed.
        events, n_events = ndimage.label(exceed_bool)
        events = _filter_min_duration(events, n_events, min_duration)
        if join_across_gaps:  # join_method == 'post-filter'
            survivors = _bridge_gaps(events > 0, max_gap)
            events, n_events = ndimage.label(survivors)

    # 4. Metrics Extraction
    mhw_duration = np.zeros(T, dtype=float) 
    mhw_category = np.zeros(T, dtype=float)
    mhw_intensity_max = np.zeros(T, dtype=float)
    mhw_intensity_cum = np.zeros(T, dtype=float)
    is_mhw = np.zeros(T, dtype=bool)

    # Re-labeling is not strictly necessary for calculation but ensures sequential IDs 
    # if some were removed during filtering. We iterate over unique surviving labels.
    active_labels = np.unique(events)
    active_labels = active_labels[active_labels != 0]

    for ev_id in active_labels:
        idx = np.where(events == ev_id)[0]
        dur = len(idx)
        # Double check duration (redundant but safe)
        if dur < min_duration: continue
        
        temps_ev = temp[idx]
        seas_ev = clim_seas[idx]
        thresh_ev = clim_thresh[idx]
        anoms = temps_ev - seas_ev
        
        # Calculate Intensity (Max and Cumulative)
        if cold_spells:
            # Max intensity is the minimum anomaly (most negative)
            i_max = np.min(anoms)
            i_cum = np.sum(anoms)
            peak_idx = np.argmin(anoms)
        else:
            i_max = np.max(anoms)
            i_cum = np.sum(anoms)
            peak_idx = np.argmax(anoms)
        
        # Categorization
        intensity_diff = thresh_ev[peak_idx] - seas_ev[peak_idx]
        if intensity_diff == 0: intensity_diff = 1e-5 
        
        ratio = anoms[peak_idx] / intensity_diff
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

def mhw_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year, max_gap_interp=2, **kwargs):
    """
    Wrapper to handle NaNs and interface with xarray.apply_ufunc, passing 
    dynamic configuration parameters to the core detection logic.

    Parameters
    ----------
    time_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1D).
    clim_start_year : int or float
        Start year of the climatology baseline.
    clim_end_year : int or float
        End year of the climatology baseline.
    max_gap_interp : int, optional
        Maximum gap length (in days) to fill via linear interpolation. 
        Gaps larger than this will remain as NaNs.
    **kwargs : optional
        Additional arguments passed directly to `detect_mhw_core`.
        (e.g., pctile, window_half_width, min_duration, cold_spells).

    Returns
    -------
    tuple
        Tuple of numpy arrays containing MHW/MCS statistics.
    """
    T = len(time_ordinal)
    
    # --- CRITICAL FIX: Dask Read-Only & Gap Handling ---
    temp = temp.copy()
    
    # 1. Fast exit for land mask (all NaNs)
    if np.isnan(temp).all():
        nan_arr = np.full(T, np.nan)
        return (nan_arr, nan_arr, nan_arr, np.zeros(T, dtype=bool), 
                nan_arr, nan_arr, nan_arr, nan_arr)

    # 2. Smart Interpolation (Only small gaps)
    if np.isnan(temp).any():
        is_valid = ~np.isnan(temp)
        valid_indices = np.flatnonzero(is_valid)

        if len(valid_indices) > 1:
            gaps = np.diff(valid_indices)
            fillable_gaps = np.where((gaps > 1) & (gaps <= (max_gap_interp + 1)))[0]

            for i in fillable_gaps:
                start_idx = valid_indices[i]
                end_idx = valid_indices[i+1]
                x_gap = np.arange(start_idx + 1, end_idx)
                temp[x_gap] = np.interp(
                    x_gap, 
                    [start_idx, end_idx], 
                    [temp[start_idx], temp[end_idx]]
                )
    
    # 3. Type safety for Numba/Core logic
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

def xrMHW_func(ds, temp_var_name, clim_period, **kwargs):
    """
    Applies the Marine Heatwave (MHW) or Marine Cold Spell (MCS) detection algorithm 
    over an entire Xarray Dataset using Dask for parallelization.

    This function acts as a high-level wrapper that prepares the data, handles 
    parallel execution via `xr.apply_ufunc`, and formats the output into a 
    clean Xarray Dataset. It implements the definition by Hobday et al. (2016).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature data and a time coordinate.
        Chunks should be optimized for time-series access (e.g., chunks={'time': -1}).
    temp_var_name : str
        The name of the temperature variable within `ds`.
    clim_period : tuple of (int, int)
        The start and end years (inclusive) for the climatology baseline (e.g., (1982, 2011)).
    **kwargs : optional
        Keyword arguments passed directly to the detection logic (`mhw_1d_wrapper` and `detect_mhw_core`).
        
        **Configuration Parameters:
        
        * **cold_spells** (bool, default: False): 
            If True, detects Marine Cold Spells (temp < threshold) instead of Heatwaves.
            
        * **pctile** (int or float, default: 90 for MHW, 10 for MCS): 
            The percentile threshold for detection.
            
        * **min_duration** (int, default: 5): 
            Minimum duration in days for an event to be recorded.
            
        * **window_half_width** (int, default: 5): 
            Half-width of the window used for climatology calculation (total window = 2*w + 1).
            
        * **smooth_pctile** (bool, default: True): 
            Whether to smooth the climatology and threshold using a moving average.
            
        * **smooth_width** (int, default: 31): 
            The width of the moving window for smoothing the climatology (if `smooth_pctile` is True).
            
        **Gap Handling Parameters:**
        
        * **max_gap_interp** (int, default: 2): 
            *Data Gap Handling:* Maximum length of consecutive missing values (NaNs) 
            to fill via linear interpolation before detection. Larger gaps remain NaNs.
            
        * **join_across_gaps** (bool, default: True): 
            *Event Gap Handling:* Whether to logically join two events separated by a 
            short return to normal temperatures.
            
        * **max_gap** (int, default: 2): 
            *Event Gap Handling:* Maximum number of days below the threshold (for MHW) 
            or above (for MCS) allowed to bridge two events into a single continuous event.

    Returns
    -------
    xr.Dataset
        A new dataset with the same spatial coordinates as the input, containing the 
        following variables (masked where no event is detected):
        
        * `{prefix}_intensity`: Temperature anomaly relative to climatology.
        * `{prefix}_duration`: Duration of the event (days).
        * `{prefix}_category`: Categorization (1=Moderate, 2=Strong, etc.).
        * `{prefix}_max_intensity`: Peak intensity during the event.
        * `{prefix}_cum_intensity`: Cumulative intensity (degree-days).
        * `climatology`: The calculated seasonal climatology (full time series).
        * `threshold`: The calculated seasonal threshold (full time series).
        
        *Note: `{prefix}` is 'mhw' by default, or 'mcs' if cold_spells=True.*

    Notes
    -----
    This function utilizes `xr.apply_ufunc` with `dask='parallelized'`, allowing 
    it to scale to datasets larger than memory and utilize cluster resources.
    """
    # 1. Prepare Time Coordinate
    # Convert time to ordinal integers for efficient numba/numpy processing
    if 'time' not in ds.indexes:
        raise ValueError("Input dataset must have a 'time' coordinate/index.")
        
    time_index = ds.indexes['time']
    # Handle different calendar types if necessary, assuming standard datetime here
    try:
        time_ordinal_np = time_index.map(lambda x: x.toordinal()).values
    except AttributeError:
        # Fallback for cftime objects or other non-standard dates if strictly needed,
        # otherwise raise error. For standard use, toordinal() is sufficient.
        raise TypeError("Time coordinate must be convertible to ordinal (datetime-like).")
    
    time_ordinal_da = xr.DataArray(
        time_ordinal_np, 
        coords={'time': ds['time']}, 
        dims='time'
    )

    # 2. Configure Arguments for apply_ufunc
    func_kwargs = {
        'clim_start_year': clim_period[0], 
        'clim_end_year': clim_period[1]
    }
    func_kwargs.update(kwargs)

    # Handle defaults logic for MCS vs MHW to ensure consistency
    if 'cold_spells' not in func_kwargs:
        func_kwargs['cold_spells'] = False
    
    # Auto-set default percentile: 90th for Heatwaves, 10th for Cold Spells
    if 'pctile' not in func_kwargs:
        func_kwargs['pctile'] = 10 if func_kwargs['cold_spells'] else 90

    # Define output types matching the return tuple of the core function
    # (seas, thresh, anomaly, is_mhw, duration, category, intensity_max, intensity_cum)
    output_dtypes = [np.float32, np.float32, np.float32, bool, np.float32, np.float32, np.float32, np.float32]
    
    # 3. Apply Vectorized UFunc
    # This maps the 1D wrapper function over every spatial pixel in parallel
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
    
    # Unpack results from the returned tuple of DataArrays
    (seas, thresh, anomaly, is_mhw, duration, category, 
     intensity_max, intensity_cum) = results

    # 4. Assemble Output Dataset
    ds_out = xr.Dataset()
    
    # Determine prefix based on event type for scientifically accurate variable names
    prefix = 'mcs' if func_kwargs['cold_spells'] else 'mhw'

    # Apply Mask: Variables are filled with NaNs where no event is detected.
    # This significantly reduces storage size and simplifies downstream plotting.
    ds_out[f'{prefix}_intensity'] = anomaly.where(is_mhw)
    ds_out[f'{prefix}_duration'] = duration.where(is_mhw)
    ds_out[f'{prefix}_category'] = category.where(is_mhw)
    ds_out[f'{prefix}_max_intensity'] = intensity_max.where(is_mhw)
    ds_out[f'{prefix}_cum_intensity'] = intensity_cum.where(is_mhw)
    
    # Climatology and Threshold exist for all time steps
    ds_out['climatology'] = seas
    ds_out['threshold'] = thresh

    # Preserve original coordinates (lat, lon, depth, etc.) from the input dataset
    ds_out = ds_out.assign_coords(ds.coords)
    
    # Copy attributes to retain metadata (units, long_name, etc.)
    ds_out.attrs = ds.attrs
    ds_out.attrs['mhw_detection_period'] = str(clim_period)
    ds_out.attrs['mhw_type'] = 'Cold Spell' if func_kwargs['cold_spells'] else 'Heatwave'
    
    return ds_out