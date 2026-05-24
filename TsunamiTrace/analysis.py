"""
TsunamiTrace.analysis — post-processing tools for ray-traced wave fields.

Functions here consume the output of trace_rays() and return derived
geophysical products: gridded travel times, arrival envelopes, etc.
"""
import numpy as np


def grid_travel_times(ray_lon, ray_lat, dt,
                      lon_arr, lat_arr, depth,
                      bin_deg=0.1, fill=True):
    """
    Grid ray-traced positions onto a regular lon/lat grid, keeping the
    minimum (first-arrival) travel time in each cell.

    Parameters
    ----------
    ray_lon : ndarray, shape (n_rays, n_steps)
        Ray longitudes in degrees, as returned by ``trace_rays()``.
        NaN after a ray terminates.
    ray_lat : ndarray, shape (n_rays, n_steps)
        Ray latitudes in degrees, as returned by ``trace_rays()``.
    dt : float
        Integration time step in seconds used in the ``trace_rays()`` call.
    lon_arr : ndarray, shape (n_lon,)
        Longitude axis of the bathymetry grid in degrees (ascending).
    lat_arr : ndarray, shape (n_lat,)
        Latitude axis of the bathymetry grid in degrees (ascending).
    depth : ndarray, shape (n_lon, n_lat)
        Bathymetry in metres, **positive = ocean** (as returned by
        ``load_bathymetry()``).  Used to build the ocean mask; land cells
        are set to NaN in the output.
    bin_deg : float, default 0.1
        Output grid cell size in degrees.  Coarser bins give denser ray
        coverage per cell and fewer empty gaps; finer bins resolve more
        spatial detail but require a denser ray fan to fill completely.
        Rule of thumb: 10–20× the bathymetry grid spacing works well.
    fill : bool, default True
        If ``True``, empty ocean bins (shadow zones or sparse ray coverage)
        are filled by linear interpolation from neighbouring filled bins.
        The fill seeds are exclusively true first-arrival values, so it
        cannot introduce a non-first-arrival into the output.
        If ``False``, empty bins are left as NaN.

    Returns
    -------
    lon_bin : ndarray, shape (n_lon_bin,)
        Longitude centres of the output grid in degrees.
    lat_bin : ndarray, shape (n_lat_bin,)
        Latitude centres of the output grid in degrees.
    travel_time : ndarray, shape (n_lat_bin, n_lon_bin)
        First-arrival travel time in **hours**.  NaN over land and, when
        ``fill=False``, in any bin that no ray entered.
        Shape follows the matplotlib row-major convention (lat × lon) so
        the three return values can be passed directly to ``contourf``::

            lon_bin, lat_bin, tt = grid_travel_times(...)
            plt.contourf(lon_bin, lat_bin, tt)

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.

    Notes
    -----
    The output grid shares the geographic extent of ``lon_arr`` / ``lat_arr``
    but at the coarser ``bin_deg`` resolution.  The ocean mask is derived
    by linearly interpolating ``depth`` onto the bin centres, so it
    accurately reflects the fine bathymetry even at coarse bin sizes.
    """
    try:
        from scipy.stats import binned_statistic_2d
        from scipy.interpolate import griddata, RegularGridInterpolator
    except ImportError:
        raise ImportError(
            "grid_travel_times requires scipy.  "
            "Install with:  pip install scipy"
        )

    # ── tag every non-NaN ray position with its elapsed time ─────────────────
    n_steps    = ray_lon.shape[1]
    step_times = np.arange(n_steps) * dt
    times_2d   = np.broadcast_to(step_times[np.newaxis, :], ray_lon.shape).copy()

    valid    = ~np.isnan(ray_lon)
    pts_lon  = ray_lon[valid]
    pts_lat  = ray_lat[valid]
    pts_time = times_2d[valid] / 3600.0   # seconds → hours

    # ── build the output bin grid ─────────────────────────────────────────────
    lon_bin       = np.arange(lon_arr[0], lon_arr[-1] + bin_deg, bin_deg)
    lat_bin       = np.arange(lat_arr[0], lat_arr[-1] + bin_deg, bin_deg)
    lon_bin_edges = np.append(lon_bin - bin_deg / 2, lon_bin[-1] + bin_deg / 2)
    lat_bin_edges = np.append(lat_bin - bin_deg / 2, lat_bin[-1] + bin_deg / 2)

    # ── ocean mask at bin resolution ──────────────────────────────────────────
    # depth is (n_lon, n_lat) — axes are (lon_arr, lat_arr).
    depth_interp = RegularGridInterpolator(
        (lon_arr, lat_arr), depth,
        method='linear', bounds_error=False, fill_value=0.0,
    )
    LON_BIN, LAT_BIN = np.meshgrid(lon_bin, lat_bin)   # each (n_lat_bin, n_lon_bin)
    ocean_mask = depth_interp((LON_BIN, LAT_BIN)) > 0

    # ── bin ray points, keep the minimum (first-arrival) time per cell ────────
    tt_bin, _, _, _ = binned_statistic_2d(
        pts_lon, pts_lat, pts_time,
        statistic='min',
        bins=[lon_bin_edges, lat_bin_edges],
    )
    # binned_statistic_2d returns (n_lon_bin, n_lat_bin); transpose to
    # (n_lat_bin, n_lon_bin) for matplotlib row-major convention.
    tt_bin = tt_bin.T

    # ── fill empty ocean bins ─────────────────────────────────────────────────
    if fill:
        good      = ~np.isnan(tt_bin)
        fill_mask = ~good & ocean_mask
        if fill_mask.any():
            tt_bin[fill_mask] = griddata(
                (LON_BIN[good], LAT_BIN[good]),
                tt_bin[good],
                (LON_BIN[fill_mask], LAT_BIN[fill_mask]),
                method='linear',
            )

    # ── mask land ─────────────────────────────────────────────────────────────
    travel_time = tt_bin.copy()
    travel_time[~ocean_mask] = np.nan

    return lon_bin, lat_bin, travel_time
