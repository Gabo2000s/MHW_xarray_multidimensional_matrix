"""
Regression tests: xrMHW  vs  marineHeatWaves (Oliver / Hobday et al. 2016)
==========================================================================

Goal
----
Prove that xrMHW, run with ``join_method='post-filter'`` (the Hobday/Oliver
default), reproduces Eric Oliver's reference ``marineHeatWaves`` implementation
to within floating-point tolerance, and that the multidimensional
``xrMHW_func`` path is consistent with the 1-D core, pixel by pixel.

Design choices (vs. the exploratory notebook)
---------------------------------------------
* EXACT agreement, not statistical "equivalence". We assert identical event
  counts, identical start/end dates and durations, and intensities/climatology
  matching within a tight tolerance. Any mismatch is reported in full so it can
  be explained, never averaged away with a Wilcoxon test.
* Deterministic SYNTHETIC data (seeded). No dependency on large external files,
  so this runs in CI in seconds. Real-data validation belongs in the docs/demo
  notebook, not the test suite.
* The single most decisive check is element-wise agreement of the seasonal
  climatology and the 90th-percentile threshold (``test_climatology_threshold``).

Test dependencies
-----------------
    pip install pytest numpy pandas scipy xarray marineHeatWaves
``marineHeatWaves`` and ``xarray`` are skipped gracefully if absent, so the
numpy-only core tests (land mask, gap filling) always run.

Run with:
    pytest -v test_against_marineHeatWaves.py
"""

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

import xrMHW

# ----------------------------------------------------------------------------
# Shared parameters and synthetic-data fixture
# ----------------------------------------------------------------------------
CLIM_PERIOD = (1991, 2020)
PARAMS = dict(
    pctile=90,
    min_duration=5,
    max_gap=2,
    window_half_width=5,
    smooth_pctile=True,
    smooth_width=31,
    join_across_gaps=True,
    join_method="post-filter",   # corrected xrMHW: default reproduces Oliver bit-for-bit
)

# Tolerances. clim/thresh need a little slack because xrMHW and Oliver handle
# Feb-29 / the 366-day grid slightly differently; everything else should be
# essentially exact.
ATOL_CLIM = 1e-6       # deg C; corrected version is bit-exact
ATOL_INTENSITY = 1e-6  # deg C; corrected version is bit-exact
MAX_DUR_DIFF = 0      # days; allow at most 1-day boundary slack, report all


@pytest.fixture(scope="module")
def synthetic():
    """30 years of daily SST: seasonal cycle + reproducible noise + injected
    warm events of known length so detection has unambiguous targets."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    temp = seasonal + rng.normal(0.0, 0.8, len(dates))

    # Inject several clean events of known duration.
    def add_event(start, length, amp):
        i = np.where(dates == np.datetime64(start))[0][0]
        temp[i:i + length] += amp

    add_event("2010-07-01", 12, 4.0)   # long, extreme
    add_event("2005-09-10", 6, 2.2)    # short, moderate/strong
    add_event("2018-02-15", 8, 3.0)    # winter event
    add_event("1998-06-05", 5, 2.5)    # exactly at the min-duration boundary

    t_ord = np.array([d.toordinal() for d in dates]).astype(int)
    return dates, t_ord, temp.astype(float)


# ----------------------------------------------------------------------------
# Helpers to put Oliver's output into the same shape as xrMHW's
# ----------------------------------------------------------------------------
def run_xrmhw_core(t_ord, temp):
    """Returns the 8-tuple from the real core:
    (seas, thresh, anomaly, is_mhw, duration, category, int_max, int_cum)."""
    return xrMHW.mhw_1d_wrapper(
        t_ord, temp.copy(), CLIM_PERIOD[0], CLIM_PERIOD[1], **PARAMS
    )


def events_from_mask(dates, is_mhw, duration, int_max, int_cum, category):
    """Collapse xrMHW's per-day masked arrays into one row per event."""
    labels, n = ndi.label(is_mhw)
    rows = []
    for k in range(1, n + 1):
        idx = np.where(labels == k)[0]
        rows.append(dict(
            start=dates[idx[0]].date(),
            end=dates[idx[-1]].date(),
            duration=int(duration[idx[0]]),
            int_max=float(int_max[idx[0]]),
            int_cum=float(int_cum[idx[0]]),
            category=int(category[idx[0]]),
        ))
    return rows


def events_from_oliver(mhw):
    """One row per Oliver event, same keys as events_from_mask."""
    cat_map = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}
    rows = []
    for i in range(mhw["n_events"]):
        cat = mhw["category"][i]
        rows.append(dict(
            start=mhw["date_start"][i],
            end=mhw["date_end"][i],
            duration=int(mhw["duration"][i]),
            int_max=float(mhw["intensity_max"][i]),
            int_cum=float(mhw["intensity_cumulative"][i]),
            category=cat_map.get(cat, cat),
        ))
    return rows


def match_by_overlap(oliver_rows, xr_rows):
    """Pair events that overlap in time. Returns matched pairs plus any
    unmatched events from either side (these are the interesting failures)."""
    pairs, used = [], set()
    for o in oliver_rows:
        for j, x in enumerate(xr_rows):
            if j in used:
                continue
            if max(o["start"], x["start"]) <= min(o["end"], x["end"]):
                pairs.append((o, x))
                used.add(j)
                break
    matched_o = {id(o) for o, _ in pairs}
    unmatched_o = [o for o in oliver_rows if id(o) not in matched_o]
    unmatched_x = [x for j, x in enumerate(xr_rows) if j not in used]
    return pairs, unmatched_o, unmatched_x


# ----------------------------------------------------------------------------
# THE decisive test: element-wise climatology & threshold
# ----------------------------------------------------------------------------
def test_climatology_threshold(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    seas_xr, thresh_xr = run_xrmhw_core(t_ord, temp)[:2]
    _, clim = mhw_mod.detect(
        t_ord, temp.copy(), climatologyPeriod=list(CLIM_PERIOD),
        minDuration=PARAMS["min_duration"], maxGap=PARAMS["max_gap"],
        windowHalfWidth=PARAMS["window_half_width"], smoothPercentile=True,
    )
    d_seas = np.nanmax(np.abs(seas_xr - clim["seas"]))
    d_thr = np.nanmax(np.abs(thresh_xr - clim["thresh"]))
    print(f"\nmax|Δ climatology| = {d_seas:.2e} °C   max|Δ threshold| = {d_thr:.2e} °C")
    assert d_seas < ATOL_CLIM, f"climatology differs by {d_seas:.2e} °C"
    assert d_thr < ATOL_CLIM, f"threshold differs by {d_thr:.2e} °C"


def test_event_count(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)

    mhw, _ = mhw_mod.detect(
        t_ord, temp.copy(), climatologyPeriod=list(CLIM_PERIOD),
        minDuration=PARAMS["min_duration"], maxGap=PARAMS["max_gap"],
        windowHalfWidth=PARAMS["window_half_width"], smoothPercentile=True,
    )
    o_rows = events_from_oliver(mhw)
    print(f"\nN events: Oliver={len(o_rows)}  xrMHW={len(xr_rows)}")
    assert len(xr_rows) == len(o_rows), (
        f"event count differs: Oliver={len(o_rows)} xrMHW={len(xr_rows)}"
    )


def test_event_dates_durations_intensities(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)

    mhw, _ = mhw_mod.detect(
        t_ord, temp.copy(), climatologyPeriod=list(CLIM_PERIOD),
        minDuration=PARAMS["min_duration"], maxGap=PARAMS["max_gap"],
        windowHalfWidth=PARAMS["window_half_width"], smoothPercentile=True,
    )
    o_rows = events_from_oliver(mhw)

    pairs, unmatched_o, unmatched_x = match_by_overlap(o_rows, xr_rows)

    # Report a full diff so any discrepancy is explainable, not hidden.
    print("\n  start(O)     dur(O) dur(xr)  imax(O)  imax(xr)  cum(O)   cum(xr)")
    dur_fail, int_fail = [], []
    for o, x in pairs:
        print(f"  {o['start']}   {o['duration']:5d} {x['duration']:6d}  "
              f"{o['int_max']:7.3f} {x['int_max']:8.3f}  "
              f"{o['int_cum']:7.2f} {x['int_cum']:8.2f}")
        if abs(o["duration"] - x["duration"]) > MAX_DUR_DIFF:
            dur_fail.append((o["start"], o["duration"], x["duration"]))
        if abs(o["int_max"] - x["int_max"]) > ATOL_INTENSITY:
            int_fail.append((o["start"], o["int_max"], x["int_max"]))

    assert not unmatched_o, f"Oliver events with no xrMHW match: {unmatched_o}"
    assert not unmatched_x, f"xrMHW events with no Oliver match: {unmatched_x}"
    assert not dur_fail, f"duration mismatch > {MAX_DUR_DIFF} d: {dur_fail}"
    assert not int_fail, f"peak intensity mismatch > {ATOL_INTENSITY}: {int_fail}"


def test_categories(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)
    mhw, _ = mhw_mod.detect(
        t_ord, temp.copy(), climatologyPeriod=list(CLIM_PERIOD),
        minDuration=PARAMS["min_duration"], maxGap=PARAMS["max_gap"],
        windowHalfWidth=PARAMS["window_half_width"], smoothPercentile=True,
    )
    o_rows = events_from_oliver(mhw)
    pairs, _, _ = match_by_overlap(o_rows, xr_rows)
    bad = [(o["start"], o["category"], x["category"])
           for o, x in pairs if o["category"] != x["category"]]
    assert not bad, f"category mismatch (start, Oliver, xrMHW): {bad}"


# ----------------------------------------------------------------------------
# Core-only tests (no Oliver, no xarray): always run
# ----------------------------------------------------------------------------
def test_land_mask_fast_exit(synthetic):
    dates, t_ord, _ = synthetic
    land = np.full(len(dates), np.nan)
    seas, thresh, anom, is_mhw, *_ = xrMHW.mhw_1d_wrapper(
        t_ord, land, CLIM_PERIOD[0], CLIM_PERIOD[1], **PARAMS
    )
    assert not np.asarray(is_mhw).any()
    assert np.isnan(seas).all() and np.isnan(thresh).all()


def test_data_gap_interpolation(synthetic):
    """A short (<= max_gap_interp) run of NaNs inside an event must be bridged
    so the event is not artificially fragmented."""
    dates, t_ord, temp = synthetic
    i0 = np.where(dates == np.datetime64("2010-07-01"))[0][0]
    gapped = temp.copy()
    gapped[i0 + 5:i0 + 7] = np.nan          # 2-day data gap inside the event
    _, _, _, is_mhw, *_ = xrMHW.mhw_1d_wrapper(
        t_ord, gapped, CLIM_PERIOD[0], CLIM_PERIOD[1],
        max_gap_interp=2, **PARAMS
    )
    block = np.asarray(is_mhw)[i0:i0 + 12]
    assert block.sum() == 12, f"event fragmented by data gap: {block.astype(int)}"


# ----------------------------------------------------------------------------
# Multidimensional consistency: xrMHW_func over a grid == 1-D core per pixel
# ----------------------------------------------------------------------------
def test_grid_matches_1d(synthetic):
    xr = pytest.importorskip("xarray")
    dates, t_ord, temp = synthetic

    # Build a tiny (time, lat, lon) cube: 3 identical ocean pixels + 1 land.
    ocean = temp
    land = np.full_like(temp, np.nan)
    cube = np.stack([
        np.stack([ocean, ocean], axis=-1),
        np.stack([ocean, land], axis=-1),
    ], axis=-1)  # (time, lat=2, lon=2); [1,1] is land
    da = xr.DataArray(
        cube, dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [0, 1], "lon": [0, 1]},
        name="sst",
    )
    ds = da.to_dataset()

    out = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD, **PARAMS)

    # 1-D reference for an ocean pixel
    _, _, _, is_mhw_1d, dur_1d, *_ = run_xrmhw_core(t_ord, temp)

    # Ocean pixel (0,0) must equal the 1-D result
    is_mhw_grid = out["mhw_duration"].isel(lat=0, lon=0).notnull().values
    assert np.array_equal(is_mhw_grid, np.asarray(is_mhw_1d)), \
        "grid ocean pixel does not match 1-D core"

    # Land pixel (1,1) must be entirely NaN / no events
    land_dur = out["mhw_duration"].isel(lat=1, lon=1).values
    assert np.isnan(land_dur).all(), "land pixel produced spurious events"
