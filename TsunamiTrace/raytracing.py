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


def _resolve_omega(period=None, frequency=None, wavelength=None,
                   local_wavelength=None, local_depth=None):
    """
    Reduce the mutually-exclusive wave-specification arguments to the single
    angular frequency ω (rad/s) conserved along every ray, or ``None`` for the
    non-dispersive default (shallow-water sqrt(g·h) everywhere).

    At most ONE wave may be specified.  The options are

        period                       ω = 2π / period
        frequency                    ω = 2π · frequency
        wavelength                   DEEP-WATER wavelength:
                                     k = 2π/λ,  ω = sqrt(g·k)
        (local_wavelength,           in-situ (measured) wavelength at a
         local_depth)                reference depth h:
                                     k = 2π/λ_local,
                                     ω = sqrt(g·k·tanh(k·h))

    ``local_wavelength`` and ``local_depth`` describe ONE option and must be
    supplied together; the deep-water ``wavelength`` and the in-situ
    ``local_wavelength`` are different physical quantities (see ``trace_rays``).
    """
    if (local_wavelength is None) != (local_depth is None):
        raise ValueError(
            "local_wavelength and local_depth must be supplied together "
            "(the in-situ wavelength and the depth at which it was measured).  "
            f"Got local_wavelength={local_wavelength}, local_depth={local_depth}."
        )
    local_pair = local_wavelength is not None

    n_given = sum(x is not None for x in (period, frequency, wavelength)) + local_pair
    if n_given > 1:
        raise ValueError(
            "Specify at most one of period, frequency, wavelength, "
            "(local_wavelength, local_depth).  "
            f"Got: period={period}, frequency={frequency}, "
            f"wavelength={wavelength}, local_wavelength={local_wavelength}, "
            f"local_depth={local_depth}."
        )

    if period is not None:
        return 2.0 * np.pi / float(period)
    if frequency is not None:
        return 2.0 * np.pi * float(frequency)
    if wavelength is not None:
        # deep-water wavelength: k = 2π/λ, ω = sqrt(g·k)
        k_deep = 2.0 * np.pi / float(wavelength)
        return np.sqrt(_G * k_deep)
    if local_pair:
        # in-situ wavelength measured at depth h: invert the full dispersion
        # relation exactly (no iteration) for ω.
        k = 2.0 * np.pi / float(local_wavelength)
        return np.sqrt(_G * k * np.tanh(k * float(local_depth)))
    return None


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
    c_phase : ndarray, same shape as depth
        Phase speed ω/k in m/s.  Zero for land cells (depth ≤ 0).
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

    return np.where(ocean, c_group, 0.0), np.where(ocean, c_phase, 0.0)


def trace_rays(lon_arr, lat_arr, depth, dt, max_time,
               source_lon, source_lat, azimuths_deg,
               period=None, frequency=None, wavelength=None,
               local_wavelength=None, local_depth=None,
               refraction="phase"):
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
        DEEP-WATER wavelength in metres.  The wavenumber ``k = 2π / λ``
        is interpreted as the deep-water value, giving
        ``ω = sqrt(g · k)`` via the deep-water dispersion relation.
        This is the natural input for source-scale arguments: a landslide
        or volcanic source with characteristic length ``L`` generates waves
        with dominant wavelength ``λ ≈ L``.  This is a DIFFERENT quantity
        from ``local_wavelength`` below and the two must not be confused.
    local_wavelength : float, optional
        In-situ (locally measured) wavelength in metres, e.g. from a spatial
        band-pass of a sea-surface-height image.  Must be given together with
        ``local_depth``.  The wavenumber ``k = 2π / λ_local`` is taken to hold
        at depth ``local_depth``, and ω is recovered from the FULL dispersion
        relation ``ω = sqrt(g · k · tanh(k · h))``.  Unlike ``wavelength`` (a
        deep-water quantity) this is the wavelength actually observed at finite
        depth; feeding a locally measured wavelength into ``wavelength`` would
        trace the wrong wave.
    local_depth : float, optional
        Depth in metres at which ``local_wavelength`` was measured.  Must be
        given together with ``local_wavelength``.
    refraction : {'phase', 'group'}, optional
        Which slowness drives the ray BENDING.  ``'phase'`` (default) is the
        physically correct choice: the path bends by the phase slowness while
        travel time still accrues at the group speed.  ``'group'`` reproduces
        the legacy single-field behaviour (bending by the group slowness) and
        exists only for controlled before/after comparisons — it matches the
        old rays bitwise.  Ignored for non-dispersive runs, where the phase and
        group speeds coincide.

    At most one of ``period``, ``frequency``, ``wavelength``,
    ``(local_wavelength, local_depth)`` may be given; the last counts as a
    single option (the two must be supplied together).  If none is given (the
    default) the shallow-water phase speed ``c = sqrt(g · h)`` is used
    everywhere — the standard tsunami approximation.

    Dispersive wave speed
    ~~~~~~~~~~~~~~~~~~~~~
    When a wave parameter is supplied both speeds of the full linear
    dispersion relation are used, for their two distinct roles:

        ω² = g · k · tanh(k · h)          (solved for k at each depth h)
        c_group = (ω/k)/2 · [1 + 2kh/sinh(2kh)]
        c_phase = ω / k

    Energy — and therefore the travel time recorded ALONG each ray — moves at
    the **group speed**, so the position equations advance at ``c_group``.  The
    ray PATH, however, bends according to the **phase slowness** (that is the
    quantity the eikonal / Snell's law is written in), so refraction is driven
    by ``1/c_phase`` with a ``c_group/c_phase`` factor from the Hamiltonian ray
    equations.  The two coincide only in the shallow-water limit (``kh → 0``);
    using the group slowness to bend (``refraction='group'``) under-refracts
    short waves and, past the c_group maximum near ``kh ≈ 1.2``, bends them the
    wrong way.

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
        If more than one wave option among ``period``, ``frequency``,
        ``wavelength``, ``(local_wavelength, local_depth)`` is given, if only
        one of ``local_wavelength`` / ``local_depth`` is supplied, or if
        ``depth.shape`` does not match ``(len(lon_arr), len(lat_arr))``.
    """
    DEG_TO_RAD = np.pi / 180.0

    lon_arr      = np.asarray(lon_arr,      dtype=float)
    lat_arr      = np.asarray(lat_arr,      dtype=float)
    depth        = np.asarray(depth,        dtype=float)
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)

    # ── wave parameter → conserved angular frequency ω ─────────────────────────
    omega = _resolve_omega(period, frequency, wavelength,
                           local_wavelength, local_depth)

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

    # ── speed fields: group (travel time) and phase (ray bending) ─────────────
    ocean_depth = np.where(depth < 0.0, 0.0, depth)   # land → 0
    ocean       = ocean_depth > 0.0

    if omega is None:
        # Non-dispersive: shallow-water c = sqrt(g·h).  Phase and group speeds
        # coincide and the refraction ratio is EXACTLY unity — assigned (not
        # divided) so non-dispersive runs stay bitwise reproducible.
        safe_depth = np.where(ocean, ocean_depth, 1.0)
        c_group    = np.sqrt(_G * safe_depth)
        c_phase    = c_group
        ratio      = np.ones_like(c_group)
    else:
        # Dispersive: the path bends by the PHASE slowness while energy (travel
        # time) advances at the GROUP speed.
        c_group, c_phase = _dispersive_group_speed(ocean_depth, omega)
        safe_cp0 = np.where(c_phase > 0.0, c_phase, 1.0)
        ratio    = np.where(ocean, c_group / safe_cp0, 1.0)   # land → 1

    # Reproducibility switch: 'group' collapses the phase field onto the group
    # field and sets ratio = 1, restoring the legacy single-field equations
    # exactly (bitwise).  'phase' is the default, physically correct behaviour.
    if refraction == "group":
        c_phase = c_group
        ratio   = np.ones_like(c_group)
    elif refraction != "phase":
        raise ValueError(
            f"refraction must be 'phase' or 'group', got {refraction!r}.")

    # Land cells get sentinel slowness = 1 to trigger the boundary exit in
    # _integrate_rays without a divide-by-zero.  Group slowness drives the
    # position equations and the land test; phase slowness drives the bending.
    safe_cg  = np.where(c_group > 0.0, c_group, 1.0)
    slowness = np.where(ocean, 1.0 / safe_cg, 1.0)            # group, 1/c_g
    safe_cp  = np.where(c_phase > 0.0, c_phase, 1.0)
    u_phase  = np.where(ocean, 1.0 / safe_cp, 1.0)            # phase, 1/c_p

    # ── phase-slowness gradients (drive refraction) ───────────────────────────
    u_phase_grad_phi   = np.diff(u_phase, axis=0) / dphi_rad
    u_phase_grad_colat = np.diff(u_phase, axis=1) / dcolat_rad

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
        slowness, u_phase, ratio,
        u_phase_grad_phi, u_phase_grad_colat,
        phi0_arr, theta0_arr, ray_dir0_arr,
        phi_grid_start, theta_grid_start,
        n_lon, n_lat, ocean_depth,
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
