"""
tests/test_rk4.py — pytest unit tests for the RK4 ray integrator.

On a uniform-depth ocean the slowness gradients are zero everywhere, so each
ray must follow a great-circle path at constant speed sqrt(g*depth).  The
tests below verify this analytically, covering:

  - Great-circle accuracy across multiple azimuths (max deviation < 1e-3°)
  - Exact symmetry: due-north ray has zero longitude drift
  - Exact symmetry: due-east equatorial ray has zero latitude drift
  - Arc-length accuracy: integrated distance matches wave_speed * t within 1 m

Tests the implementation of the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.
"""
import numpy as np
import pytest
from TsunamiTrace._rungekutta import _integrate_ray

# ── physical constants ────────────────────────────────────────────────────────
DEG_TO_RAD   = np.pi / 180.0
EARTH_RADIUS = 6_371_000.0   # m
G            = 9.8           # m/s²
DEPTH        = 4_000.0       # m — flat ocean
WAVE_SPEED   = np.sqrt(G * DEPTH)
SLOWNESS     = 1.0 / WAVE_SPEED

# ── integration parameters ────────────────────────────────────────────────────
GRID_SPACING_DEG = 0.1
DT               = 10.0     # s
MAX_TIME         = 3_600.0  # s — 1 hour; ray travels ~6.4° on Earth's surface
SOURCE_LON       = 0.0
SOURCE_LAT       = 0.0


# ── shared fixture ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def flat_ocean():
    """
    20° × 20° uniform-depth grid centred on the equator/prime-meridian.
    All slowness gradients are exactly zero.
    """
    lon_arr = np.arange(-10.0, 10.0 + GRID_SPACING_DEG, GRID_SPACING_DEG)
    lat_arr = np.arange(-10.0, 10.0 + GRID_SPACING_DEG, GRID_SPACING_DEG)
    n_lon, n_lat = len(lon_arr), len(lat_arr)

    return dict(
        n_lon         = n_lon,
        n_lat         = n_lat,
        slowness_grid = np.full((n_lon, n_lat), SLOWNESS),
        depth_grid    = np.full((n_lon, n_lat), DEPTH),
        grad_phi      = np.zeros((n_lon - 1, n_lat)),
        grad_colat    = np.zeros((n_lon, n_lat - 1)),
        dphi_rad      = GRID_SPACING_DEG * DEG_TO_RAD,
        dcolat_rad    = GRID_SPACING_DEG * DEG_TO_RAD,
        time_arr      = np.arange(0.0, MAX_TIME + DT, DT),
        source_ix     = int(np.argmin(np.abs(lon_arr - SOURCE_LON))),
        source_iy     = int(np.argmin(np.abs(lat_arr - SOURCE_LAT))),
    )


# ── helper ────────────────────────────────────────────────────────────────────
def _run_ray(grid, azimuth):
    """Run _integrate_ray for a given azimuth on the flat-ocean grid."""
    g = grid
    return _integrate_ray(
        g['time_arr'], DT, g['dphi_rad'], g['dcolat_rad'],
        g['slowness_grid'], g['grad_phi'], g['grad_colat'],
        azimuth, SOURCE_LON, SOURCE_LAT,
        g['source_ix'], g['source_iy'], g['n_lon'], g['n_lat'], g['depth_grid'],
    )


def _great_circle(azimuth_deg, t):
    """
    Analytical great-circle position (lat_deg, lon_deg) at times t.
    Uses the spherical law of cosines for colatitude and the four-parts
    formula for the longitude increment.
    """
    theta_0 = (90.0 - SOURCE_LAT) * DEG_TO_RAD
    phi_0   = SOURCE_LON * DEG_TO_RAD
    z_0     = azimuth_deg * DEG_TO_RAD
    alpha   = WAVE_SPEED * t / EARTH_RADIUS   # arc angle

    cos_theta = np.clip(
        np.cos(theta_0) * np.cos(alpha)
        + np.sin(theta_0) * np.sin(alpha) * np.cos(z_0),
        -1.0, 1.0,
    )
    theta_gc  = np.arccos(cos_theta)
    delta_phi = np.arctan2(
        np.sin(alpha) * np.sin(z_0),
        np.cos(alpha) * np.sin(theta_0) - np.cos(theta_0) * np.sin(alpha) * np.cos(z_0),
    )
    return 90.0 - theta_gc / DEG_TO_RAD, (phi_0 + delta_phi) / DEG_TO_RAD


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_great_circle_accuracy(flat_ocean, azimuth):
    """
    Integrated ray path must stay within 1e-3° of the analytical great circle
    at every time step.  This bounds the RK4 truncation error for a 10 s
    time step over a 1-hour integration on a 4000 m flat ocean.
    """
    phi_ray, theta_ray, *_ = _run_ray(flat_ocean, azimuth)
    t_ray = np.arange(len(phi_ray)) * DT

    lat_gc, lon_gc = _great_circle(azimuth, t_ray)
    lon_ray = phi_ray / DEG_TO_RAD
    lat_ray = 90.0 - theta_ray / DEG_TO_RAD

    deviation = np.sqrt((lon_ray - lon_gc)**2 + (lat_ray - lat_gc)**2)
    assert deviation.max() < 1e-3, (
        f"azimuth {azimuth}°: max deviation {deviation.max():.2e}° exceeds 1e-3°"
    )


def test_northward_ray_longitude_constant(flat_ocean):
    """
    A due-north ray (azimuth=0°) has dphi/dt = 0 exactly.
    Longitude must not drift from the source value at all.
    """
    phi_ray, *_ = _run_ray(flat_ocean, 0.0)
    lon_drift = np.abs(phi_ray / DEG_TO_RAD - SOURCE_LON).max()
    assert lon_drift < 1e-10, (
        f"Due-north ray: longitude drift {lon_drift:.2e}° (expected exactly 0)"
    )


def test_eastward_equatorial_ray_latitude_constant(flat_ocean):
    """
    A due-east ray (azimuth=90°) from the equator has dtheta/dt = 0 exactly
    because sin(theta)=1 and the spherical correction term vanishes.
    Latitude must not drift from 0°.
    """
    _, theta_ray, *_ = _run_ray(flat_ocean, 90.0)
    lat_drift = np.abs(90.0 - theta_ray / DEG_TO_RAD - SOURCE_LAT).max()
    assert lat_drift < 1e-10, (
        f"Due-east equatorial ray: latitude drift {lat_drift:.2e}° (expected exactly 0)"
    )


def test_arc_length_matches_wave_speed(flat_ocean):
    """
    On a flat ocean the ray travels at exactly sqrt(g*depth).
    The spherical arc length from source to each recorded position must match
    wave_speed * t to within 1 m over a 1-hour integration.
    """
    phi_ray, theta_ray, *_ = _run_ray(flat_ocean, 45.0)
    t_ray = np.arange(len(phi_ray)) * DT

    theta_0 = (90.0 - SOURCE_LAT) * DEG_TO_RAD
    phi_0   = SOURCE_LON * DEG_TO_RAD
    cos_arc = np.clip(
        np.cos(theta_ray) * np.cos(theta_0)
        + np.sin(theta_ray) * np.sin(theta_0) * np.cos(phi_ray - phi_0),
        -1.0, 1.0,
    )
    arc_m     = np.arccos(cos_arc) * EARTH_RADIUS
    expected_m = WAVE_SPEED * t_ray
    max_err_m  = np.abs(arc_m - expected_m).max()
    assert max_err_m < 1.0, (
        f"Arc length error {max_err_m:.2f} m exceeds 1 m tolerance"
    )
