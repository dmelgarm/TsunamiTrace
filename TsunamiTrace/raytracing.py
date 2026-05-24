"""
Tsunami ray-path integrator — fans rays from one or more sources.

Implements the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohora, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.

Originally implemented in MATLAB; this is the Python port.
"""
import numpy as np
from ._rungekutta import _integrate_rays


def trace_rays(lon_arr, lat_arr, depth, dt, max_time,
               source_lon, source_lat, azimuths_deg):
    """
    Trace tsunami rays through a bathymetry grid.

    Fans out one ray per entry in ``azimuths_deg`` from each source location,
    integrates each path with the 4th-order Runge-Kutta ray-tracing equations,
    and returns the lon/lat history of every ray.  All rays (across all sources
    and azimuths) are advanced simultaneously in a single vectorised RK4 pass.

    Parameters
    ----------
    lon_arr : array_like, shape (n_lon,)
        Longitude coordinates of the grid columns in degrees, uniformly spaced
        and ascending.
    lat_arr : array_like, shape (n_lat,)
        Latitude coordinates of the grid rows in degrees, uniformly spaced
        and ascending.
    depth : ndarray, shape (n_lon, n_lat)
        Bathymetry in metres.  Positive = ocean, negative or zero = land.
        First axis is longitude (lon_arr), second is latitude (lat_arr).
        For matplotlib contour plots use ``plt.contour(lon_arr, lat_arr, depth.T)``
        to transpose to the [lat, lon] row-major layout contour expects.
        Cells with depth < 0 are zeroed internally before computing slowness;
        the caller's array is not modified.
    dt : float
        Integration time step in seconds.
    max_time : float
        Maximum integration time in seconds.
    source_lon : float or array_like, shape (n_sources,)
        Source longitude(s) in degrees.  Pass a scalar for a single source
        (backward-compatible) or a 1-D array for a multi-source / finite-fault
        ensemble.
    source_lat : float or array_like, shape (n_sources,)
        Source latitude(s) in degrees.  Must have the same length as
        ``source_lon`` when an array is supplied.
    azimuths_deg : array_like, shape (n_azimuths,)
        Initial ray azimuths in degrees: 0 = north, 90 = east, 180 = south,
        270 = west (clockwise from north).  The same fan of azimuths is
        launched from every source.

    Returns
    -------
    ray_lon_deg : ndarray
        Longitude of each ray at each recorded state, in degrees.

        - **Scalar source** (float input) — shape ``(n_azimuths, n_steps)``.
        - **Array source** — shape ``(n_sources, n_azimuths, n_steps)``.

        Positions after a ray terminates (grid boundary, shallow water, or
        land) are NaN.
    ray_lat_deg : ndarray
        Latitude of each ray at each recorded state, in degrees.
        Same shape as ``ray_lon_deg``.  NaN after termination.
    ray_dir_deg : ndarray
        ODE ray direction in degrees.  Same shape as ``ray_lon_deg``.
        NaN after termination.  Column 0 holds the initial ODE angle
        derived from each azimuth: ``(180 - azimuth_deg) % 360``.

    Notes
    -----
    Slowness field
        Tsunami wave speed ``c = sqrt(g * depth)``; slowness ``n = 1/c`` with
        ``g = 9.8 m/s²``.  Land/dry cells (depth <= 0) are assigned ``n = 1``,
        which triggers the land early-exit condition in ``_integrate_rays``.

    Signed colatitude step
        ``dcolat_rad`` is signed: when ``lat_arr`` is ascending (the standard
        convention), colatitude decreases as the array index increases, so
        ``dcolat_rad < 0``.  This sign is critical for correct grid-cell index
        snapping and gradient direction — stripping it reverses the effective
        latitude axis and mirrors all ray paths.

    Output column count
        ``n_steps = len(time_arr) + 1``.  Column 0 = initial condition
        (t = 0); column k = state after k completed RK4 steps (t = k*dt).

    Multi-source layout
        When ``source_lon``/``source_lat`` are arrays the ray layout within
        the integrator is ``[src0_az0, src0_az1, …, src1_az0, …]``.  The
        returned arrays are reshaped to ``(n_sources, n_azimuths, n_steps)``
        before returning.  Passing these to ``grid_travel_times`` uses
        ``binned_statistic_2d(statistic='min')`` internally, so the minimum
        travel time across all sources is kept automatically.
    """
    DEG_TO_RAD = np.pi / 180.0
    G          = 9.8              # gravitational acceleration, m/s²

    lon_arr      = np.asarray(lon_arr,      dtype=float)
    lat_arr      = np.asarray(lat_arr,      dtype=float)
    depth        = np.asarray(depth,        dtype=float)
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)

    # Detect scalar vs array source — preserve backward compatibility
    scalar_source  = np.ndim(source_lon) == 0
    source_lon_arr = np.atleast_1d(np.asarray(source_lon, dtype=float))
    source_lat_arr = np.atleast_1d(np.asarray(source_lat, dtype=float))
    n_sources  = len(source_lon_arr)
    n_azimuths = len(azimuths_deg)

    # ── sanity check ─────────────────────────────────────────────────────────
    expected_shape = (len(lon_arr), len(lat_arr))
    if depth.shape != expected_shape:
        raise ValueError(
            f"depth shape {depth.shape} does not match "
            f"(len(lon_arr), len(lat_arr)) = {expected_shape}. "
            "First axis must be longitude, second must be latitude."
        )

    # ── colatitude and grid spacing ───────────────────────────────────────────
    # dcolat_rad is SIGNED: ascending lat_arr means decreasing colatitude, so
    # dcolat_rad < 0.  The sign governs grid-cell index computation and the
    # direction of the slowness gradient along the latitude axis.
    colat_arr  = 90.0 - lat_arr
    dcolat_rad = (colat_arr[1] - colat_arr[0]) * DEG_TO_RAD   # signed
    dphi_rad   = abs(lon_arr[1] - lon_arr[0])  * DEG_TO_RAD   # always positive

    n_lon, n_lat = depth.shape

    # ── slowness field ────────────────────────────────────────────────────────
    # Land cells (depth <= 0) get sentinel value n = 1 to trigger the boundary
    # check in _integrate_rays without causing a divide-by-zero.
    local_depth = np.where(depth < 0.0, 0.0, depth)
    safe_depth  = np.where(local_depth > 0.0, local_depth, 1.0)
    slowness    = np.where(
        local_depth > 0.0,
        1.0 / np.sqrt(G * safe_depth),
        1.0,                             # land sentinel
    )

    # ── slowness gradients ────────────────────────────────────────────────────
    slowness_grad_phi   = np.diff(slowness, axis=0) / dphi_rad    # (n_lon-1, n_lat)
    slowness_grad_colat = np.diff(slowness, axis=1) / dcolat_rad  # (n_lon, n_lat-1)

    # ── time array ────────────────────────────────────────────────────────────
    time_arr = np.arange(0.0, max_time + dt, dt)

    # ── initial conditions — n_sources * n_azimuths total rays ───────────────
    # Layout: [src0_az0, src0_az1, …, src0_azN, src1_az0, …, srcM_azN]
    phi0_arr     = np.repeat(source_lon_arr * DEG_TO_RAD, n_azimuths)
    theta0_arr   = np.repeat((90.0 - source_lat_arr) * DEG_TO_RAD, n_azimuths)
    ray_dir0_arr = np.tile(((180.0 - azimuths_deg) % 360.0) * DEG_TO_RAD, n_sources)

    # Grid reference points for absolute grid-cell index computation
    phi_grid_start   = lon_arr[0] * DEG_TO_RAD
    theta_grid_start = (90.0 - lat_arr[0]) * DEG_TO_RAD

    # ── ray integration — all rays advanced simultaneously ───────────────────
    out_phi, out_theta, out_ray_dir = _integrate_rays(
        time_arr, dt, dphi_rad, dcolat_rad,
        slowness, slowness_grad_phi, slowness_grad_colat,
        phi0_arr, theta0_arr, ray_dir0_arr,
        phi_grid_start, theta_grid_start,
        n_lon, n_lat, local_depth,
    )

    # ── convert to geographic coordinates ────────────────────────────────────
    ray_lon_deg = out_phi     / DEG_TO_RAD
    ray_lat_deg = 90.0 - out_theta / DEG_TO_RAD
    ray_dir_deg = out_ray_dir / DEG_TO_RAD

    # ── reshape for multi-source output ──────────────────────────────────────
    # Scalar source: keep (n_azimuths, n_steps) — backward compatible.
    # Array source:  reshape to (n_sources, n_azimuths, n_steps).
    if not scalar_source:
        n_steps     = ray_lon_deg.shape[1]
        ray_lon_deg = ray_lon_deg.reshape(n_sources, n_azimuths, n_steps)
        ray_lat_deg = ray_lat_deg.reshape(n_sources, n_azimuths, n_steps)
        ray_dir_deg = ray_dir_deg.reshape(n_sources, n_azimuths, n_steps)

    return ray_lon_deg, ray_lat_deg, ray_dir_deg
