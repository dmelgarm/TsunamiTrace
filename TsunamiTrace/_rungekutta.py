"""
Vectorised RK4 integrator for tsunami ray tracing.

Implements the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.

Originally implemented in MATLAB; this is the Python port.
"""
import numpy as np

_EARTH_RADIUS = 6_371_000.0   # metres
_DEG_TO_RAD   = np.pi / 180.0


# ── ODE right-hand sides ──────────────────────────────────────────────────────
# Module-level so they are not re-created inside the integration loop.

def _dtheta(theta, ray_dir, n, R):
    return np.cos(ray_dir) / (n * R)


def _dphi(theta, ray_dir, n, R):
    # sin(theta) in the denominator: faster longitudinal drift near the poles
    return np.sin(ray_dir) / (n * R * np.sin(theta))


def _ddir(theta, ray_dir, n, dn_dcolat, dn_dphi, R):
    # First two terms: Snell's-law refraction from the slowness gradient.
    # Third term: spherical correction for meridian convergence (cot theta).
    sin_d = np.sin(ray_dir)
    cos_d = np.cos(ray_dir)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    return (
        -sin_d * dn_dcolat / (n**2 * R)
        + cos_d * dn_dphi  / (n**2 * R * sin_t)
        - sin_d * cos_t    / (n * R * sin_t)
    )


# ── vectorised integrator ─────────────────────────────────────────────────────

def _integrate_rays(time_arr, dt, dphi_rad, dcolat_rad,
                    slowness, slowness_grad_phi, slowness_grad_colat,
                    azimuths_deg, source_lon, source_lat,
                    source_ix, source_iy, n_lon, n_lat, depth):
    """
    Integrate all tsunami rays simultaneously using vectorised RK4.

    All rays share the same bathymetry grid and are advanced together at each
    time step.  A boolean mask eliminates terminated rays from further
    computation without padding their output arrays.

    Parameters
    ----------
    time_arr : ndarray, shape (N,)
        Time array in seconds, evenly spaced with step dt.
    dt : float
        Time step in seconds.
    dphi_rad : float
        Grid spacing in the longitude (phi) direction, radians.
    dcolat_rad : float
        Grid spacing in the colatitude (theta) direction, radians.
    slowness : ndarray, shape (n_lon, n_lat)
        Slowness 1/sqrt(g*depth) in s/m.  Land cells pre-set to >= 1.
    slowness_grad_phi : ndarray, shape (n_lon-1, n_lat)
        dn/dphi in s/(m·rad), from np.diff(slowness, axis=0) / dphi_rad.
    slowness_grad_colat : ndarray, shape (n_lon, n_lat-1)
        dn/dtheta in s/(m·rad), from np.diff(slowness, axis=1) / dcolat_rad.
    azimuths_deg : array_like, shape (n_rays,)
        Initial ray azimuths in degrees (0=N, 90=E, 180=S, 270=W).
    source_lon, source_lat : float
        Source position in degrees.
    source_ix, source_iy : int
        Nearest grid indices to the source.
    n_lon, n_lat : int
        Grid dimensions.
    depth : ndarray, shape (n_lon, n_lat)
        Bathymetry in metres (positive = water).

    Returns
    -------
    out_phi : ndarray, shape (n_rays, n_steps)
        Ray longitude in radians.  NaN after each ray terminates.
    out_theta : ndarray, shape (n_rays, n_steps)
        Ray colatitude in radians.  NaN after termination.
    out_ray_dir : ndarray, shape (n_rays, n_steps)
        Ray direction in radians.  NaN after termination.

    Notes
    -----
    Equations of motion — Snell's law on a sphere:

        dtheta/dt   = cos(ray_dir) / (n * R)
        dphi/dt     = sin(ray_dir) / (n * R * sin(theta))
        dray_dir/dt = -sin(ray_dir) * dn/dtheta / (n^2 * R)
                      + cos(ray_dir) * dn/dphi   / (n^2 * R * sin(theta))
                      - sin(ray_dir) * cos(theta) / (n * R * sin(theta))

    Termination fires when any of these conditions holds for a given ray:
      1. Grid boundary: lon_idx outside [0, n_lon-2] or lat_idx outside [0, n_lat-2].
      2. Shallow water or land: depth < 10 m.
      3. Land sentinel: n_here >= 1.

    Slowness and its gradients are frozen at the grid cell corresponding to
    the ray's position at the start of each time step and held constant
    through all four RK stages (piecewise-constant medium per cell).
    """
    R      = _EARTH_RADIUS
    n_rays = len(azimuths_deg)
    n_steps = len(time_arr) + 1

    # Pre-allocate output arrays — NaN marks positions after termination
    out_phi     = np.full((n_rays, n_steps), np.nan)
    out_theta   = np.full((n_rays, n_steps), np.nan)
    out_ray_dir = np.full((n_rays, n_steps), np.nan)

    # Initial conditions — all rays start from the same source point
    phi0   = source_lon * _DEG_TO_RAD
    theta0 = (90.0 - source_lat) * _DEG_TO_RAD

    phi     = np.full(n_rays, phi0)
    theta   = np.full(n_rays, theta0)
    # Convert geographic azimuth (CW from N) to ODE angle (CCW from +e_theta):
    # z = pi - azimuth  →  z=0 points south, z=pi points north
    ray_dir = (180.0 - np.asarray(azimuths_deg, dtype=float)) * _DEG_TO_RAD

    out_phi[:, 0]     = phi
    out_theta[:, 0]   = theta
    out_ray_dir[:, 0] = ray_dir

    alive = np.ones(n_rays, dtype=bool)

    for step in range(len(time_arr)):
        if not alive.any():
            break

        # Snap current position to nearest grid cell
        lon_idx = source_ix + np.round((phi   - phi0)   / dphi_rad).astype(int)
        lat_idx = source_iy + np.round((theta - theta0) / dcolat_rad).astype(int)

        # Clip to valid index ranges before array access —
        # dead rays may have wandered outside the grid.
        ix    = np.clip(lon_idx, 0, n_lon - 2)   # slowness_grad_phi  axis-0 limit
        iy    = np.clip(lat_idx, 0, n_lat - 1)
        iy_gc = np.clip(lat_idx, 0, n_lat - 2)   # slowness_grad_colat axis-1 limit

        # Freeze local medium at the current grid cell for all four RK stages
        n_here         = slowness[ix, iy]
        dn_dphi_here   = slowness_grad_phi[ix, iy]
        dn_dcolat_here = slowness_grad_colat[ix, iy_gc]

        # ── RK4 ──────────────────────────────────────────────────────────────
        k1_t = _dtheta(theta, ray_dir, n_here, R)
        k1_p = _dphi(  theta, ray_dir, n_here, R)
        k1_d = _ddir(  theta, ray_dir, n_here, dn_dcolat_here, dn_dphi_here, R)

        t2 = theta   + 0.5 * dt * k1_t
        d2 = ray_dir + 0.5 * dt * k1_d
        k2_t = _dtheta(t2, d2, n_here, R)
        k2_p = _dphi(  t2, d2, n_here, R)
        k2_d = _ddir(  t2, d2, n_here, dn_dcolat_here, dn_dphi_here, R)

        t3 = theta   + 0.5 * dt * k2_t
        d3 = ray_dir + 0.5 * dt * k2_d
        k3_t = _dtheta(t3, d3, n_here, R)
        k3_p = _dphi(  t3, d3, n_here, R)
        k3_d = _ddir(  t3, d3, n_here, dn_dcolat_here, dn_dphi_here, R)

        t4 = theta   + dt * k3_t
        d4 = ray_dir + dt * k3_d
        k4_t = _dtheta(t4, d4, n_here, R)
        k4_p = _dphi(  t4, d4, n_here, R)
        k4_d = _ddir(  t4, d4, n_here, dn_dcolat_here, dn_dphi_here, R)

        # Simpson-weighted update
        dt6 = dt / 6.0
        new_theta   = theta   + (k1_t + 2*k2_t + 2*k3_t + k4_t) * dt6
        new_phi     = phi     + (k1_p + 2*k2_p + 2*k3_p + k4_p) * dt6
        new_ray_dir = ray_dir + (k1_d + 2*k2_d + 2*k3_d + k4_d) * dt6

        # Apply update only to rays still alive
        theta   = np.where(alive, new_theta,   theta)
        phi     = np.where(alive, new_phi,     phi)
        ray_dir = np.where(alive, new_ray_dir, ray_dir)

        # Record new state before applying termination
        out_phi[alive, step + 1]     = phi[alive]
        out_theta[alive, step + 1]   = theta[alive]
        out_ray_dir[alive, step + 1] = ray_dir[alive]

        # Termination checks
        out_of_bounds = (
            (lon_idx < 0) | (lon_idx > n_lon - 2) |
            (lat_idx < 0) | (lat_idx > n_lat - 2)
        )
        ix_d    = np.clip(lon_idx, 0, n_lon - 1)
        iy_d    = np.clip(lat_idx, 0, n_lat - 1)
        shallow = depth[ix_d, iy_d] < 10.0
        land    = n_here >= 1.0

        alive &= ~(out_of_bounds | shallow | land)

    return out_phi, out_theta, out_ray_dir
