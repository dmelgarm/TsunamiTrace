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


def _ddir(theta, ray_dir, n_group, u_phase, ratio, du_dcolat, du_dphi, R):
    # First two terms: Snell's-law refraction, driven by the PHASE slowness
    # gradient with the c_group/c_phase weight from the Hamiltonian ray eqs.
    # Third term: spherical meridian-convergence correction (cot theta); it is
    # a geometric term and correctly keeps the GROUP slowness.
    sin_d = np.sin(ray_dir)
    cos_d = np.cos(ray_dir)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    return (
        -sin_d * ratio * du_dcolat / (u_phase**2 * R)
        + cos_d * ratio * du_dphi  / (u_phase**2 * R * sin_t)
        - sin_d * cos_t / (n_group * R * sin_t)
    )


# ── vectorised integrator ─────────────────────────────────────────────────────

def _integrate_rays(time_arr, dt, dphi_rad, dcolat_rad,
                    slowness, u_phase, ratio,
                    u_phase_grad_phi, u_phase_grad_colat,
                    phi0_arr, theta0_arr, ray_dir0_arr,
                    phi_grid_start, theta_grid_start,
                    n_lon, n_lat, depth):
    """
    Integrate all tsunami rays simultaneously using vectorised RK4.

    All rays share the same bathymetry grid and are advanced together at each
    time step.  A boolean mask eliminates terminated rays from further
    computation without padding their output arrays.

    Initial conditions are passed in as pre-built arrays, making this function
    agnostic to whether the rays originate from one source or many.

    Parameters
    ----------
    time_arr : ndarray, shape (N,)
        Time array in seconds, evenly spaced with step dt.
    dt : float
        Time step in seconds.
    dphi_rad : float
        Grid spacing in the longitude (phi) direction, radians. Always positive.
    dcolat_rad : float
        Signed grid spacing in the colatitude (theta) direction, radians.
        Negative when lat_arr is ascending (the standard convention), because
        colatitude decreases as the lat index increases.
    slowness : ndarray, shape (n_lon, n_lat)
        Group slowness 1/c_group in s/m.  Land cells pre-set to >= 1.  Drives
        the position equations and the land-termination test.
    u_phase : ndarray, shape (n_lon, n_lat)
        Phase slowness 1/c_phase in s/m.  Land cells pre-set to >= 1.  Drives
        the refraction terms.
    ratio : ndarray, shape (n_lon, n_lat)
        c_group/c_phase (1 for non-dispersive / 'group' mode / land).
    u_phase_grad_phi : ndarray, shape (n_lon-1, n_lat)
        d(u_phase)/dphi in s/(m*rad).
    u_phase_grad_colat : ndarray, shape (n_lon, n_lat-1)
        d(u_phase)/dtheta in s/(m*rad).
    phi0_arr : ndarray, shape (n_rays,)
        Initial longitude of every ray in radians.
    theta0_arr : ndarray, shape (n_rays,)
        Initial colatitude of every ray in radians.
    ray_dir0_arr : ndarray, shape (n_rays,)
        Initial ODE ray direction of every ray in radians.
    phi_grid_start : float
        Longitude of grid index 0 in radians (lon_arr[0] * DEG_TO_RAD).
    theta_grid_start : float
        Colatitude of grid index 0 in radians ((90 - lat_arr[0]) * DEG_TO_RAD).
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
    """
    R      = _EARTH_RADIUS
    n_rays  = len(phi0_arr)
    n_steps = len(time_arr) + 1

    # Pre-allocate output arrays — NaN marks positions after termination
    out_phi     = np.full((n_rays, n_steps), np.nan)
    out_theta   = np.full((n_rays, n_steps), np.nan)
    out_ray_dir = np.full((n_rays, n_steps), np.nan)

    phi     = phi0_arr.copy()
    theta   = theta0_arr.copy()
    ray_dir = ray_dir0_arr.copy()

    out_phi[:, 0]     = phi
    out_theta[:, 0]   = theta
    out_ray_dir[:, 0] = ray_dir

    alive = np.ones(n_rays, dtype=bool)

    for step in range(len(time_arr)):
        if not alive.any():
            break

        # Absolute grid-cell index from current ray position.
        # dphi_rad > 0 always; dcolat_rad < 0 for ascending lat_arr.
        lon_idx = np.round((phi   - phi_grid_start)   / dphi_rad).astype(int)
        lat_idx = np.round((theta - theta_grid_start) / dcolat_rad).astype(int)

        # Clip to valid index ranges before array access —
        # dead rays may have wandered outside the grid.
        ix    = np.clip(lon_idx, 0, n_lon - 2)   # slowness_grad_phi  axis-0 limit
        iy    = np.clip(lat_idx, 0, n_lat - 1)
        iy_gc = np.clip(lat_idx, 0, n_lat - 2)   # slowness_grad_colat axis-1 limit

        # Freeze local medium at the current grid cell for all four RK stages.
        # Group slowness drives position + the land test; phase slowness (with
        # the c_g/c_p ratio and its gradients) drives the refraction in _ddir.
        n_here         = slowness[ix, iy]
        u_here         = u_phase[ix, iy]
        ratio_here     = ratio[ix, iy]
        du_dphi_here   = u_phase_grad_phi[ix, iy]
        du_dcolat_here = u_phase_grad_colat[ix, iy_gc]

        # ── RK4 ──────────────────────────────────────────────────────────────
        k1_t = _dtheta(theta, ray_dir, n_here, R)
        k1_p = _dphi(  theta, ray_dir, n_here, R)
        k1_d = _ddir(  theta, ray_dir, n_here, u_here, ratio_here,
                       du_dcolat_here, du_dphi_here, R)

        t2 = theta   + 0.5 * dt * k1_t
        d2 = ray_dir + 0.5 * dt * k1_d
        k2_t = _dtheta(t2, d2, n_here, R)
        k2_p = _dphi(  t2, d2, n_here, R)
        k2_d = _ddir(  t2, d2, n_here, u_here, ratio_here,
                       du_dcolat_here, du_dphi_here, R)

        t3 = theta   + 0.5 * dt * k2_t
        d3 = ray_dir + 0.5 * dt * k2_d
        k3_t = _dtheta(t3, d3, n_here, R)
        k3_p = _dphi(  t3, d3, n_here, R)
        k3_d = _ddir(  t3, d3, n_here, u_here, ratio_here,
                       du_dcolat_here, du_dphi_here, R)

        t4 = theta   + dt * k3_t
        d4 = ray_dir + dt * k3_d
        k4_t = _dtheta(t4, d4, n_here, R)
        k4_p = _dphi(  t4, d4, n_here, R)
        k4_d = _ddir(  t4, d4, n_here, u_here, ratio_here,
                       du_dcolat_here, du_dphi_here, R)

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
