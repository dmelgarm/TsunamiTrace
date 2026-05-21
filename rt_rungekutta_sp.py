"""
Single-ray RK4 integrator for spherical tsunami ray tracing.

Implements the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.

Originally implemented in MATLAB; this is the Python port.
"""
import numpy as np


def rt_rungekutta_sp(time_arr, dt, dphi_rad, dcolat_rad,
                     slowness, slowness_grad_phi, slowness_grad_colat,
                     initial_azimuth_deg, source_lon, source_lat,
                     source_ix, source_iy, n_lon, n_lat, depth):
    """
    Integrate a single tsunami ray on a sphere using 4th-order Runge-Kutta.

    The state vector is (theta, phi, ray_dir) where theta is colatitude, phi
    is longitude (both in radians), and ray_dir is the propagation direction
    in radians.  Slowness and its spatial gradients are frozen from the grid
    cell at the start of each time step and held constant through all four
    RK stages, consistent with a piecewise-constant medium across each cell.

    Parameters
    ----------
    time_arr : array_like, shape (N,)
        Time array in seconds, evenly spaced with step dt.  Typically built
        with ``np.arange(0, max_time + dt, dt)``.
    dt : float
        Time step in seconds.
    dphi_rad : float
        Grid spacing in the longitude (phi) direction, in radians.
    dcolat_rad : float
        Grid spacing in the colatitude (theta) direction, in radians.
    slowness : ndarray, shape (n_lon, n_lat)
        Slowness field 1/sqrt(g * depth) in s/m, indexed [lon_idx, lat_idx].
        Land cells must be pre-set to a sentinel >= 1 so that the
        ``n_here >= 1`` termination check fires correctly.
    slowness_grad_phi : ndarray, shape (n_lon-1, n_lat)
        Slowness gradient along the longitude axis (dn/dphi), s/(m*rad).
        Built from ``np.diff(slowness, axis=0) / dphi_rad``.
    slowness_grad_colat : ndarray, shape (n_lon, n_lat-1)
        Slowness gradient along the colatitude axis (dn/dtheta), s/(m*rad).
        Built from ``np.diff(slowness, axis=1) / dcolat_rad``.
    initial_azimuth_deg : float
        Initial ray azimuth in degrees: 0 = north, 90 = east, 180 = south,
        270 = west (clockwise from north).
        Internally converted to the ODE angle (CCW from +e_theta) via
        z = (180 - azimuth_deg) * pi/180.
    source_lon : float
        Source longitude in degrees.
    source_lat : float
        Source latitude in degrees.
    source_ix : int
        Source grid index along the longitude axis, 0-based.
    source_iy : int
        Source grid index along the colatitude axis, 0-based.
    n_lon : int
        Grid size along the longitude axis (first dimension of slowness).
    n_lat : int
        Grid size along the colatitude axis (second dimension of slowness).
    depth : ndarray, shape (n_lon, n_lat)
        Bathymetry in metres (positive = ocean depth), indexed [lon_idx, lat_idx].

    Returns
    -------
    phi : ndarray of float
        Ray longitude in radians at each recorded state.  At most
        len(time_arr) + 1 values (initial condition plus one per completed
        step); shorter if the ray terminates early.
    theta : ndarray of float
        Ray colatitude in radians, same length as phi.
    ray_dir : ndarray of float
        Ray direction in radians, same length as phi.
    ix_hist : ndarray of int
        Grid longitude index (0-based) recorded after each RK step.
        At most len(time_arr) values; shorter on early exit.
    iy_hist : ndarray of int
        Grid colatitude index (0-based), same length as ix_hist.

    Notes
    -----
    Equations of motion — spherical ray-tracing (Snell's law on a sphere):

        dtheta/dt   = cos(ray_dir) / (n * R)
        dphi/dt     = sin(ray_dir) / (n * R * sin(theta))
        dray_dir/dt = -sin(ray_dir) * dn/dtheta / (n^2 * R)
                      + cos(ray_dir) * dn/dphi   / (n^2 * R * sin(theta))
                      - sin(ray_dir) * cos(theta) / (n * R * sin(theta))

    The third term is the spherical correction for meridian convergence
    (equivalent to -sin(ray_dir) * cot(theta) / (n * R)).

    Integration terminates early when any of these conditions is met:
      1. Grid boundary: lon_idx outside [0, n_lon-2] or
         lat_idx outside [0, n_lat-2].
         (slowness_grad_phi has shape (n_lon-1, n_lat), so n_lon-2 is the
          last valid lon index; slowness_grad_colat has shape (n_lon, n_lat-1),
          so n_lat-2 is the last valid lat index.)
      2. Shallow water / land: depth[lon_idx, lat_idx] < 10 m.
      3. Degenerate slowness: n_here >= 1 (land sentinel set by the caller).
    """
    DEG_TO_RAD   = np.pi / 180.0
    EARTH_RADIUS = 6_371_000.0    # metres

    # -------------------------------------------------------------------------
    # Initial conditions stored as lists that grow one element per step.
    # -------------------------------------------------------------------------
    phi     = [source_lon * DEG_TO_RAD]           # longitude, rad
    theta   = [(90.0 - source_lat) * DEG_TO_RAD]  # colatitude = 90° - latitude, rad
    # Convert geographic azimuth (clockwise from north) to the ODE's internal
    # angle (CCW from +e_theta, the southward unit vector):
    #   z_internal = pi - azimuth_geographic
    # At z=0 the ray moves southward (dtheta/dt > 0); at z=pi, northward.
    ray_dir = [(180.0 - initial_azimuth_deg) * DEG_TO_RAD]

    ix_hist: list[int] = []
    iy_hist: list[int] = []

    # lon_idx / lat_idx track the current grid cell; updated after each RK step.
    lon_idx = source_ix
    lat_idx = source_iy

    for i in range(len(time_arr)):

        # ---------------------------------------------------------------------
        # Freeze slowness and its gradients at the current cell for all four
        # RK stages.  This is equivalent to assuming a locally uniform medium
        # within each grid cell.
        # ---------------------------------------------------------------------
        n_here         = slowness[lon_idx, lat_idx]
        dn_dphi_here   = slowness_grad_phi[lon_idx, lat_idx]
        dn_dcolat_here = slowness_grad_colat[lon_idx, lat_idx]

        # ---------------------------------------------------------------------
        # ODE right-hand sides.
        # phi does not appear explicitly in any equation, so only
        # (theta_v, dir_v) are needed as arguments.
        # ---------------------------------------------------------------------

        def colat_rate(theta_v, dir_v):
            # dtheta/dt: colatitude changes at cos(ray_dir) / (n * R)
            return np.cos(dir_v) / (n_here * EARTH_RADIUS)

        def lon_rate(theta_v, dir_v):
            # dphi/dt: sin(theta) in the denominator causes faster longitudinal
            # drift near the poles where meridians converge
            return np.sin(dir_v) / (n_here * EARTH_RADIUS * np.sin(theta_v))

        def dir_rate(theta_v, dir_v):
            # dray_dir/dt: first two terms are Snell's law refraction due to
            # the slowness gradient; third term is the spherical correction
            # for meridian convergence (cot(theta) written as cos/sin)
            return (
                -np.sin(dir_v) * dn_dcolat_here / (n_here**2 * EARTH_RADIUS)
                + np.cos(dir_v) * dn_dphi_here / (n_here**2 * EARTH_RADIUS * np.sin(theta_v))
                - np.sin(dir_v) * np.cos(theta_v) / (n_here * EARTH_RADIUS * np.sin(theta_v))
            )

        # ---------------------------------------------------------------------
        # RK4 — stage 1: slopes at the start of the interval
        # ---------------------------------------------------------------------
        dtheta_1 = colat_rate(theta[i], ray_dir[i])
        dphi_1   = lon_rate(  theta[i], ray_dir[i])
        ddir_1   = dir_rate(  theta[i], ray_dir[i])

        # ---------------------------------------------------------------------
        # RK4 — stage 2: midpoint estimate using stage-1 slopes
        # ---------------------------------------------------------------------
        theta_s2 = theta[i]   + 0.5 * dt * dtheta_1
        dir_s2   = ray_dir[i] + 0.5 * dt * ddir_1
        dtheta_2 = colat_rate(theta_s2, dir_s2)
        dphi_2   = lon_rate(  theta_s2, dir_s2)
        ddir_2   = dir_rate(  theta_s2, dir_s2)

        # ---------------------------------------------------------------------
        # RK4 — stage 3: second midpoint estimate using stage-2 slopes
        # ---------------------------------------------------------------------
        theta_s3 = theta[i]   + 0.5 * dt * dtheta_2
        dir_s3   = ray_dir[i] + 0.5 * dt * ddir_2
        dtheta_3 = colat_rate(theta_s3, dir_s3)
        dphi_3   = lon_rate(  theta_s3, dir_s3)
        ddir_3   = dir_rate(  theta_s3, dir_s3)

        # ---------------------------------------------------------------------
        # RK4 — stage 4: endpoint estimate using stage-3 slopes
        # ---------------------------------------------------------------------
        theta_s4 = theta[i]   + dt * dtheta_3
        dir_s4   = ray_dir[i] + dt * ddir_3
        dtheta_4 = colat_rate(theta_s4, dir_s4)
        dphi_4   = lon_rate(  theta_s4, dir_s4)
        ddir_4   = dir_rate(  theta_s4, dir_s4)

        # ---------------------------------------------------------------------
        # Advance state: Simpson-weighted average of the four slope estimates
        # ---------------------------------------------------------------------
        theta.append(
            theta[i] + (dtheta_1 + 2*dtheta_2 + 2*dtheta_3 + dtheta_4) * dt / 6.0
        )
        phi.append(
            phi[i] + (dphi_1 + 2*dphi_2 + 2*dphi_3 + dphi_4) * dt / 6.0
        )
        ray_dir.append(
            ray_dir[i] + (ddir_1 + 2*ddir_2 + 2*ddir_3 + ddir_4) * dt / 6.0
        )

        # ---------------------------------------------------------------------
        # Update grid index by measuring cumulative angular displacement from
        # the source position and snapping to the nearest grid cell.
        # ---------------------------------------------------------------------
        lon_idx = source_ix + int(round((phi[i]   - phi[0])   / dphi_rad))
        lat_idx = source_iy + int(round((theta[i] - theta[0]) / dcolat_rad))
        ix_hist.append(lon_idx)
        iy_hist.append(lat_idx)

        # ---------------------------------------------------------------------
        # Termination checks.
        # ---------------------------------------------------------------------

        # 1. Grid boundary — gradient arrays are one cell shorter on each axis,
        #    so the last usable index is n_lon-2 (lon) and n_lat-2 (lat).
        if lon_idx < 0 or lon_idx > n_lon - 2 or lat_idx < 0 or lat_idx > n_lat - 2:
            break

        # 2. Shallow water or land.
        if depth[lon_idx, lat_idx] < 10:
            break

        # 3. Degenerate slowness (land sentinel or stalled ray).
        if n_here >= 1:
            break

    return (
        np.array(phi),
        np.array(theta),
        np.array(ray_dir),
        np.array(ix_hist, dtype=int),
        np.array(iy_hist, dtype=int),
    )
