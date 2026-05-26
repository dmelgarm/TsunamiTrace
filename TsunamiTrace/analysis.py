"""
TsunamiTrace.analysis — post-processing tools for ray-traced wave fields.

Functions here consume the output of trace_rays() and return derived
geophysical products: gridded travel times, arrival envelopes, etc.
"""
import numpy as np

_DEG_TO_RAD = np.pi / 180.0


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


def sample_travel_times(lon_bin, lat_bin, travel_time, lons, lats,
                        max_snap_bins=5):
    """
    Sample a travel-time grid at arbitrary lon/lat receiver locations.

    For each query point the nearest grid bin is looked up.  If that bin is
    NaN (the point landed on a land cell or an unfilled shadow zone), the
    function searches an expanding square neighbourhood and returns the value
    of the nearest non-NaN bin.  The nearest bin is chosen by Euclidean
    distance in index space, so the result is always the geographically
    closest valid sample rather than the minimum in the patch.

    Parameters
    ----------
    lon_bin : ndarray, shape (n_lon_bin,)
        Longitude bin centres, as returned by ``grid_travel_times()``.
    lat_bin : ndarray, shape (n_lat_bin,)
        Latitude bin centres, as returned by ``grid_travel_times()``.
    travel_time : ndarray, shape (n_lat_bin, n_lon_bin)
        Travel-time grid in hours, as returned by ``grid_travel_times()``.
        NaN over land.
    lons : array-like, shape (n_pts,)
        Receiver longitudes in degrees.
    lats : array-like, shape (n_pts,)
        Receiver latitudes in degrees.
    max_snap_bins : int, default 5
        Maximum search radius in grid bins when snapping a land-cell hit to
        the nearest ocean bin.  At ``bin_deg=0.1°`` the default of 5 bins
        corresponds to ~55 km.  Points still NaN after the search are
        returned as NaN.

    Returns
    -------
    times : ndarray, shape (n_pts,)
        Travel times in hours at each receiver location.  NaN for any point
        that could not be resolved within ``max_snap_bins``.
    n_snapped : int
        Number of points that were snapped from a land bin to the nearest
        ocean bin.  Useful for a quick sanity check; a large number suggests
        the receiver coordinates are significantly inland.

    Examples
    --------
    >>> lon_bin, lat_bin, tt = tt.grid_travel_times(ray_lon, ray_lat, ...)
    >>> times, n_snap = tt.sample_travel_times(
    ...     lon_bin, lat_bin, tt,
    ...     lons=dart_lons, lats=dart_lats,
    ... )
    >>> print(times * 60)   # minutes
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    # Nearest bin index for each receiver (vectorised)
    i_lons = np.argmin(np.abs(lon_bin[:, None] - lons[None, :]), axis=0)
    i_lats = np.argmin(np.abs(lat_bin[:, None] - lats[None, :]), axis=0)
    result  = travel_time[i_lats, i_lons].copy()

    n_snapped = 0
    for k in np.where(np.isnan(result))[0]:
        # Expand outward one ring at a time; pick the nearest non-NaN bin
        # by index-space distance so we don't accidentally jump far away.
        found = False
        for r in range(1, max_snap_bins + 1):
            r0 = max(0, i_lats[k] - r);  r1 = min(len(lat_bin), i_lats[k] + r + 1)
            c0 = max(0, i_lons[k] - r);  c1 = min(len(lon_bin), i_lons[k] + r + 1)
            patch = travel_time[r0:r1, c0:c1]
            valid  = ~np.isnan(patch)
            if valid.any():
                # Among valid cells in the patch find the one closest in
                # index space to the original query point.
                rows, cols = np.where(valid)
                abs_rows   = rows + r0
                abs_cols   = cols + c0
                dists      = (abs_rows - i_lats[k])**2 + (abs_cols - i_lons[k])**2
                best       = np.argmin(dists)
                result[k]  = travel_time[abs_rows[best], abs_cols[best]]
                n_snapped += 1
                found = True
                break
        # If still not found after max_snap_bins, result[k] stays NaN

    return result, n_snapped


def grid_azimuths(source_lon, source_lat, lon_bin, lat_bin):
    """
    Great-circle bearing from a source point to every cell in a bin grid.

    Useful for diagnosing which azimuths have sparse ray coverage: overlay
    this map on the raw travel-time map (``fill=False``) to read off the
    azimuths of shadow-zone bins and add targeted rays there.

    The bearing is the *initial* azimuth of the great-circle arc from the
    source to each bin centre — the direction a ray leaving the source would
    need to be aimed to reach that cell directly.

    Parameters
    ----------
    source_lon : float
        Source longitude in degrees.
    source_lat : float
        Source latitude in degrees.
    lon_bin : ndarray, shape (n_lon_bin,)
        Longitude centres of the bin grid in degrees, as returned by
        ``grid_travel_times()``.
    lat_bin : ndarray, shape (n_lat_bin,)
        Latitude centres of the bin grid in degrees, as returned by
        ``grid_travel_times()``.

    Returns
    -------
    azimuth : ndarray, shape (n_lat_bin, n_lon_bin)
        Great-circle bearing from the source to each bin centre, in degrees
        clockwise from north, range [0, 360).  Shape follows the matplotlib
        row-major convention so the array can be passed directly to
        ``pcolormesh`` or ``contourf`` alongside ``lon_bin`` and ``lat_bin``.

    Notes
    -----
    Uses the standard spherical bearing formula::

        y = sin(Δlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) − sin(lat1) * cos(lat2) * cos(Δlon)
        bearing = atan2(y, x)  [converted to degrees, wrapped to 0–360]

    This is the great-circle *initial* bearing, not the rhumb-line bearing.

    Examples
    --------
    >>> lon_b, lat_b, tt = tt.grid_travel_times(ray_lon, ray_lat, ...)
    >>> az = tt.grid_azimuths(source_lon, source_lat, lon_b, lat_b)
    >>> plt.pcolormesh(lon_b, lat_b, az, cmap='twilight', shading='nearest')
    """
    lon_bin = np.asarray(lon_bin, dtype=float)
    lat_bin = np.asarray(lat_bin, dtype=float)

    phi1 = source_lat * _DEG_TO_RAD
    lam1 = source_lon * _DEG_TO_RAD

    LON_BIN, LAT_BIN = np.meshgrid(lon_bin, lat_bin)   # (n_lat_bin, n_lon_bin)

    phi2 = LAT_BIN * _DEG_TO_RAD
    d_lam = (LON_BIN - source_lon) * _DEG_TO_RAD

    y = np.sin(d_lam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lam)

    return np.degrees(np.arctan2(y, x)) % 360.0
