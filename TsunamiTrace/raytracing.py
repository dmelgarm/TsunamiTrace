"""
Tsunami ray-path integrator — fans rays from a point source.

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

    Fans out one ray per entry in azimuths_deg, integrates each path with the
    4th-order Runge-Kutta ray-tracing equations, and returns the lon/lat
    history of every ray as 2-D arrays.

    Parameters
    ----------
    lon_arr : array_like, shape (n_lon,)
        Longitude coordinates of the grid columns in degrees, uniformly spaced.
    lat_arr : array_like, shape (n_lat,)
        Latitude coordinates of the grid rows in degrees, uniformly spaced.
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
    source_lon : float
        Source longitude in degrees.
    source_lat : float
        Source latitude in degrees.
    azimuths_deg : array_like, shape (n_rays,)
        Initial ray azimuths in degrees: 0 = north, 90 = east, 180 = south,
        270 = west (clockwise from north).

    Returns
    -------
    ray_lon_deg : ndarray, shape (n_rays, n_steps)
        Longitude of each ray at each recorded state, in degrees.
        Column k corresponds to time k * dt seconds after the source.
        Positions after a ray terminates are NaN.
    ray_lat_deg : ndarray, shape (n_rays, n_steps)
        Latitude of each ray at each recorded state, in degrees.
        NaN after termination.
    ray_dir_deg : ndarray, shape (n_rays, n_steps)
        Ray direction (ODE internal angle) in degrees.  NaN after termination.
        Column 0 holds the initial ODE angle: ``(180 - azimuth_deg) % 360``.

    Notes
    -----
    Grid spacing
        Longitude and latitude must have the same uniform spacing.
        dcolat_rad = dphi_rad = |lat_arr[1] - lat_arr[0]| * pi/180.

    Slowness
        Tsunami wave speed c = sqrt(g * depth); slowness n = 1/c with g = 9.8 m/s².
        Land/dry cells (depth <= 0) are assigned n = 1, which triggers the
        n_here >= 1 early-exit condition in _integrate_ray.

    Output column count
        n_steps = len(time_arr) + 1.  The integrator produces one state per
        completed step plus the initial condition, so column 0 = t=0 and
        column k = state after k steps (t = k*dt).

    NaN pre-fill
        Output arrays are pre-filled with NaN; positions beyond a ray's
        termination point are left as NaN rather than zero, so no valid
        position at lon=0° or lat=90° can be incorrectly masked.
    """
    DEG_TO_RAD = np.pi / 180.0
    G          = 9.8              # gravitational acceleration, m/s²

    lon_arr      = np.asarray(lon_arr,      dtype=float)
    lat_arr      = np.asarray(lat_arr,      dtype=float)
    depth        = np.asarray(depth,        dtype=float)
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)

    # ── sanity check ─────────────────────────────────────────────────────────
    expected_shape = (len(lon_arr), len(lat_arr))
    if depth.shape != expected_shape:
        raise ValueError(
            f"depth shape {depth.shape} does not match "
            f"(len(lon_arr), len(lat_arr)) = {expected_shape}. "
            "First axis must be longitude, second must be latitude."
        )

    # ── colatitude array and grid spacing ────────────────────────────────────
    # The ODE is expressed in colatitude (theta = 90° - latitude).
    # dcolat_rad must be SIGNED: when lat_arr is ascending, colatitude decreases
    # as the array index increases, so dcolat_rad is negative.  The sign is used
    # inside _integrate_rays when snapping the ray state to a grid cell index and
    # when evaluating slowness gradients — stripping the sign reverses the
    # effective latitude direction and produces mirrored/out-of-bounds rays.
    colat_arr  = 90.0 - lat_arr
    dcolat_rad = (colat_arr[1] - colat_arr[0]) * DEG_TO_RAD   # signed
    dphi_rad   = abs(lon_arr[1] - lon_arr[0])  * DEG_TO_RAD   # always positive

    n_lon, n_lat = depth.shape

    # ── source grid indices (0-based) ─────────────────────────────────────────
    source_ix = int(np.argmin(np.abs(lon_arr - source_lon)))
    source_iy = int(np.argmin(np.abs(lat_arr - source_lat)))

    # ── slowness field ────────────────────────────────────────────────────────
    # Tsunami wave speed c = sqrt(g * depth); slowness n = 1/c.
    # Land cells (depth <= 0) get sentinel value n = 1 to trigger the boundary
    # check in _integrate_ray without causing a divide-by-zero.
    local_depth = np.where(depth < 0.0, 0.0, depth)
    # Replace zero-depth cells in the denominator with 1 before dividing to
    # avoid a runtime warning; the np.where mask discards those values anyway.
    safe_depth = np.where(local_depth > 0.0, local_depth, 1.0)
    slowness   = np.where(
        local_depth > 0.0,
        1.0 / np.sqrt(G * safe_depth),
        1.0,                             # land sentinel
    )

    # ── slowness gradients ────────────────────────────────────────────────────
    # First-order finite differences along each axis; the gradient arrays are
    # one cell shorter than the slowness field on the differenced axis.
    slowness_grad_phi   = np.diff(slowness, axis=0) / dphi_rad    # (n_lon-1, n_lat)
    slowness_grad_colat = np.diff(slowness, axis=1) / dcolat_rad  # (n_lon, n_lat-1)

    # ── time array ────────────────────────────────────────────────────────────
    time_arr = np.arange(0.0, max_time + dt, dt)

    # One state per completed step plus the initial condition
    n_steps = len(time_arr) + 1
    n_rays  = len(azimuths_deg)

    # ── ray integration — all rays advanced simultaneously ───────────────────
    out_phi, out_theta, out_ray_dir = _integrate_rays(
        time_arr, dt, dphi_rad, dcolat_rad,
        slowness, slowness_grad_phi, slowness_grad_colat,
        azimuths_deg, source_lon, source_lat,
        source_ix, source_iy, n_lon, n_lat, local_depth,
    )

    ray_lon_deg = out_phi     / DEG_TO_RAD
    ray_lat_deg = 90.0 - out_theta / DEG_TO_RAD
    ray_dir_deg = out_ray_dir / DEG_TO_RAD

    return ray_lon_deg, ray_lat_deg, ray_dir_deg
