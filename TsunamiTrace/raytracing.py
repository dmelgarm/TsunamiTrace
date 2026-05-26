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

_G = 9.8   # gravitational acceleration, m/s²


def _dispersive_group_speed(depth, omega, n_iter=15):
    """
    Compute the dispersive group speed at every grid cell.

    Solves the implicit dispersion relation

        ω² = g · k · tanh(k · h)

    for the wavenumber k at each depth h using vectorised Newton–Raphson,
    then returns the group speed

        c_group = (ω/k) / 2 · [1 + 2kh / sinh(2kh)]

    Physical background
    -------------------
    Frequency ω is conserved as a wave propagates over varying bathymetry;
    wavenumber k is not.  For a given ω the dispersion relation fixes a unique
    k at each depth, and hence a unique group speed.  In the shallow-water
    limit (kh → 0) this recovers c_group = sqrt(g·h); in the deep-water
    limit (kh → ∞) it gives c_group = sqrt(g/k)/2.

    Numerics
    --------
    Newton–Raphson starting from the shallow-water guess k₀ = ω/sqrt(g·h):

        F(k)  = ω² − g·k·tanh(kh)
        F′(k) = −g · [tanh(kh) + kh/cosh²(kh)]
        k ← k − F/F′,  then clamped to k > 0

    The residual F(k) is monotone decreasing in k and the initial guess is an
    overestimate (shallow-water c > dispersive c), guaranteeing one-sided
    convergence.  Fifteen iterations drive the residual below machine
    precision for any physically realistic kh.

    Overflow safeguards: kh is clipped at 500 before evaluation of cosh and
    sinh (float64 overflows at ~710).  At those depths tanh(kh) → 1 and
    kh/cosh²(kh) → 0, recovering the exact deep-water Newton step
    k ← k − (ω² − gk)/(−g); and sinh(2kh) → ∞ so the bracket factor
    → 1, giving c_group = c_phase/2, also exact.

    Parameters
    ----------
    depth : ndarray
        Water depth in metres, shape (n_lon, n_lat).  Positive = ocean,
        zero or negative = land.
    omega : float
        Angular frequency in rad/s (conserved along rays).
    n_iter : int
        Newton–Raphson iterations (default 15; rarely needs changing).

    Returns
    -------
    c_group : ndarray, same shape as depth
        Group speed in m/s.  Zero for land cells (depth ≤ 0).
    """
    ocean = depth > 0.0
    h     = np.where(ocean, depth, 1.0)   # dummy depth on land avoids /0

    # ── Newton–Raphson ────────────────────────────────────────────────────────
    k = omega / np.sqrt(_G * h)           # shallow-water initial guess

    for _ in range(n_iter):
        kh      = np.minimum(k * h, 500.0)          # clip to prevent overflow
        tanh_kh = np.tanh(kh)
        # sech²(kh) = 1/cosh²(kh) → 0 for large kh (deep-water limit).
        # cosh(kh)² overflows float64 for kh ≳ 355, so we compute it only
        # where kh < 355 using indexed assignment — np.where would evaluate
        # both branches and trigger the overflow even for clipped kh.
        sech2   = np.zeros_like(kh)
        m       = kh < 355.0
        sech2[m] = 1.0 / np.cosh(kh[m])**2
        F       = omega**2 - _G * k * tanh_kh
        dF      = -_G * (tanh_kh + k * h * sech2)
        k       = k - F / dF
        k       = np.maximum(k, 1e-10)              # keep positive

    # ── group speed ───────────────────────────────────────────────────────────
    kh      = k * h
    c_phase = omega / k
    two_kh  = 2.0 * kh

    # 2kh/sinh(2kh): limit → 1 as kh → 0 (L'Hôpital), → 0 as kh → ∞.
    # Use the shallow-water limit directly for very small two_kh to avoid 0/0,
    # and clip before sinh to prevent overflow.
    ratio   = np.where(
        two_kh < 1e-6,
        1.0,                                         # shallow-water limit
        two_kh / np.sinh(np.minimum(two_kh, 500.0)),
    )
    c_group = 0.5 * c_phase * (1.0 + ratio)

    return np.where(ocean, c_group, 0.0)


def trace_rays(lon_arr, lat_arr, depth, dt, max_time,
               source_lon, source_lat, azimuths_deg,
               period=None, frequency=None, wavelength=None):
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
    dt : float
        Integration time step in seconds.
    max_time : float
        Maximum integration time in seconds.
    source_lon : float or array_like, shape (n_sources,)
        Source longitude(s) in degrees.  Scalar for a single source;
        1-D array for a multi-source / finite-fault ensemble.
    source_lat : float or array_like, shape (n_sources,)
        Source latitude(s) in degrees.  Same length as ``source_lon``.
    azimuths_deg : array_like, shape (n_azimuths,)
        Initial ray azimuths in degrees: 0 = north, 90 = east, 180 = south,
        270 = west (clockwise from north).
    period : float, optional
        Wave period in seconds.  Sets the angular frequency
        ``ω = 2π / period`` which is conserved along all rays.
    frequency : float, optional
        Wave frequency in Hz.  Sets ``ω = 2π · frequency``.
    wavelength : float, optional
        Deep-water wavelength in metres.  The wavenumber ``k = 2π / λ``
        is interpreted as the deep-water value, giving
        ``ω = sqrt(g · k)`` via the deep-water dispersion relation.
        This is the natural input for source-scale arguments: a landslide
        or volcanic source with characteristic length ``L`` generates waves
        with dominant wavelength ``λ ≈ L``.

    At most one of ``period``, ``frequency``, ``wavelength`` may be given.
    If none is given (the default) the shallow-water phase speed
    ``c = sqrt(g · h)`` is used everywhere — the standard tsunami
    approximation.

    Dispersive wave speed
    ~~~~~~~~~~~~~~~~~~~~~
    When a wave parameter is supplied the wave speed at each grid cell is
    the **group speed** from the full linear dispersion relation:

        ω² = g · k · tanh(k · h)   (solved for k at each depth h)
        c_group = (ω/k)/2 · [1 + 2kh/sinh(2kh)]

    Group speed carries wave energy and is the physically correct quantity
    for ray theory.  It equals ``sqrt(g·h)`` in the shallow-water limit
    (``kh → 0``) and ``sqrt(g/k)/2`` in the deep-water limit (``kh → ∞``).

    For long-period tsunamis (T > 20 min) the correction relative to
    ``sqrt(g·h)`` is less than 1 %.  For dispersive wavetrains
    (T = 2–10 min) it can reach 5–25 %, making the parameter worthwhile.

    Returns
    -------
    ray_lon_deg : ndarray
        Longitude of each ray at each recorded state, in degrees.

        - **Scalar source** — shape ``(n_azimuths, n_steps)``.
        - **Array source** — shape ``(n_sources, n_azimuths, n_steps)``.

        Positions after a ray terminates are NaN.
    ray_lat_deg : ndarray
        Latitude of each ray.  Same shape as ``ray_lon_deg``.
    ray_dir_deg : ndarray
        ODE ray direction in degrees.  Same shape as ``ray_lon_deg``.

    Raises
    ------
    ValueError
        If more than one of ``period``, ``frequency``, ``wavelength`` is
        given, or if ``depth.shape`` does not match
        ``(len(lon_arr), len(lat_arr))``.
    """
    DEG_TO_RAD = np.pi / 180.0

    lon_arr      = np.asarray(lon_arr,      dtype=float)
    lat_arr      = np.asarray(lat_arr,      dtype=float)
    depth        = np.asarray(depth,        dtype=float)
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)

    # ── wave parameter validation ─────────────────────────────────────────────
    n_given = sum(x is not None for x in (period, frequency, wavelength))
    if n_given > 1:
        raise ValueError(
            "Specify at most one of period, frequency, wavelength.  "
            f"Got: period={period}, frequency={frequency}, wavelength={wavelength}."
        )

    # Convert to angular frequency ω (conserved along rays).
    # wavelength is interpreted as the deep-water wavelength:
    #   k = 2π/λ  →  ω = sqrt(g·k)  (deep-water dispersion relation).
    if period is not None:
        omega = 2.0 * np.pi / float(period)
    elif frequency is not None:
        omega = 2.0 * np.pi * float(frequency)
    elif wavelength is not None:
        k_deep = 2.0 * np.pi / float(wavelength)
        omega  = np.sqrt(_G * k_deep)
    else:
        omega = None                             # non-dispersive (default)

    # ── detect scalar vs array source ────────────────────────────────────────
    scalar_source  = np.ndim(source_lon) == 0
    source_lon_arr = np.atleast_1d(np.asarray(source_lon, dtype=float))
    source_lat_arr = np.atleast_1d(np.asarray(source_lat, dtype=float))
    n_sources  = len(source_lon_arr)
    n_azimuths = len(azimuths_deg)

    # ── sanity check ──────────────────────────────────────────────────────────
    expected_shape = (len(lon_arr), len(lat_arr))
    if depth.shape != expected_shape:
        raise ValueError(
            f"depth shape {depth.shape} does not match "
            f"(len(lon_arr), len(lat_arr)) = {expected_shape}. "
            "First axis must be longitude, second must be latitude."
        )

    # ── grid spacing ──────────────────────────────────────────────────────────
    # dcolat_rad is SIGNED: ascending lat_arr → decreasing colatitude → negative.
    colat_arr  = 90.0 - lat_arr
    dcolat_rad = (colat_arr[1] - colat_arr[0]) * DEG_TO_RAD
    dphi_rad   = abs(lon_arr[1] - lon_arr[0])  * DEG_TO_RAD

    n_lon, n_lat = depth.shape

    # ── slowness field ────────────────────────────────────────────────────────
    local_depth = np.where(depth < 0.0, 0.0, depth)   # land → 0

    if omega is None:
        # Standard shallow-water phase speed: c = sqrt(g·h)
        safe_depth = np.where(local_depth > 0.0, local_depth, 1.0)
        c_field    = np.sqrt(_G * safe_depth)
    else:
        # Dispersive group speed from the full dispersion relation
        c_field = _dispersive_group_speed(local_depth, omega)

    # Land cells get sentinel slowness = 1 to trigger the boundary exit in
    # _integrate_rays without causing a divide-by-zero.
    safe_c   = np.where(c_field > 0.0, c_field, 1.0)
    slowness = np.where(local_depth > 0.0, 1.0 / safe_c, 1.0)

    # ── slowness gradients ────────────────────────────────────────────────────
    slowness_grad_phi   = np.diff(slowness, axis=0) / dphi_rad
    slowness_grad_colat = np.diff(slowness, axis=1) / dcolat_rad

    # ── time array ────────────────────────────────────────────────────────────
    time_arr = np.arange(0.0, max_time + dt, dt)

    # ── initial conditions ────────────────────────────────────────────────────
    phi0_arr     = np.repeat(source_lon_arr * DEG_TO_RAD, n_azimuths)
    theta0_arr   = np.repeat((90.0 - source_lat_arr) * DEG_TO_RAD, n_azimuths)
    ray_dir0_arr = np.tile(((180.0 - azimuths_deg) % 360.0) * DEG_TO_RAD, n_sources)

    phi_grid_start   = lon_arr[0] * DEG_TO_RAD
    theta_grid_start = (90.0 - lat_arr[0]) * DEG_TO_RAD

    # ── ray integration ───────────────────────────────────────────────────────
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
    if not scalar_source:
        n_steps     = ray_lon_deg.shape[1]
        ray_lon_deg = ray_lon_deg.reshape(n_sources, n_azimuths, n_steps)
        ray_lat_deg = ray_lat_deg.reshape(n_sources, n_azimuths, n_steps)
        ray_dir_deg = ray_dir_deg.reshape(n_sources, n_azimuths, n_steps)

    return ray_lon_deg, ray_lat_deg, ray_dir_deg
