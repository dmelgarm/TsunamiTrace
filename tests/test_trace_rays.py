"""
tests/test_trace_rays.py — pytest integration tests for trace_rays().

Tests the full trace_rays() pipeline on two synthetic bathymetries:

  flat_ocean : uniform 4000 m depth — used for shape/contract tests
  ridge      : Gaussian N-S submarine ridge — used for Snell's law,
               NaN consistency, and azimuthal symmetry tests

Tests the implementation of the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.
"""
import numpy as np
import pytest
import TsunamiTrace as tt

# ── constants ─────────────────────────────────────────────────────────────────
DEG_TO_RAD   = np.pi / 180.0
EARTH_RADIUS = 6_371_000.0   # m
G            = 9.8           # m/s²
DT           = 30.0          # s — time step used across all tests


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flat_ocean():
    """Uniform 4000 m ocean on a small grid for shape and contract tests."""
    lon_arr = np.linspace(130.0, 165.0, 100)
    lat_arr = np.linspace(-15.0,  15.0,  80)
    depth   = np.full((100, 80), 4_000.0)
    return dict(
        lon_arr    = lon_arr,
        lat_arr    = lat_arr,
        depth      = depth,
        wave_speed = np.sqrt(G * 4_000.0),
        source_lon = 157.0,
        source_lat = 0.0,
    )


@pytest.fixture(scope="module")
def ridge():
    """
    Gaussian N-S submarine ridge sitting between source and western boundary.
    Source at 157°E; ridge crest at 147°E, depth 400 m (wave speed ~63 m/s)
    vs deep ocean at 5000 m (wave speed ~221 m/s).
    The ridge Gaussian is symmetric about the equator, enabling symmetry tests.
    """
    N_LON, N_LAT = 150, 100
    lon_arr = np.linspace(130.0, 165.0, N_LON)
    lat_arr = np.linspace(-15.0,  15.0, N_LAT)
    LON, LAT = np.meshgrid(lon_arr, lat_arr)

    DEEP_OCEAN = 5_000.0
    RIDGE_SILL = 400.0
    RIDGE_LON  = 147.0

    ridge_rise = (DEEP_OCEAN - RIDGE_SILL) * np.exp(
        -(LON - RIDGE_LON)**2 / (2 * 1.5**2)
        - LAT**2              / (2 * 30.0**2)
    )
    depth = (DEEP_OCEAN - ridge_rise).T   # (n_lon, n_lat) for trace_rays

    return dict(
        lon_arr    = lon_arr,
        lat_arr    = lat_arr,
        depth      = depth,
        deep_ocean = DEEP_OCEAN,
        ridge_sill = RIDGE_SILL,
        source_lon = 157.0,
        source_lat = 0.0,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

def test_output_shape(flat_ocean):
    """
    Output arrays must have shape (n_rays, n_steps) where
    n_steps = len(np.arange(0, max_time + dt, dt)) + 1.
    """
    f        = flat_ocean
    N_RAYS   = 8
    MAX_TIME = 300.0
    azimuths = np.linspace(0, 315, N_RAYS)

    ray_lon, ray_lat, ray_dir = tt.trace_rays(
        f['lon_arr'], f['lat_arr'], f['depth'],
        DT, MAX_TIME, f['source_lon'], f['source_lat'], azimuths,
    )

    n_steps = len(np.arange(0.0, MAX_TIME + DT, DT)) + 1
    assert ray_lon.shape == (N_RAYS, n_steps)
    assert ray_lat.shape == (N_RAYS, n_steps)
    assert ray_dir.shape == (N_RAYS, n_steps)


def test_invalid_depth_shape(flat_ocean):
    """Passing a depth array whose shape doesn't match lon/lat must raise ValueError."""
    f = flat_ocean
    with pytest.raises(ValueError):
        tt.trace_rays(
            f['lon_arr'], f['lat_arr'], np.ones((10, 10)),
            DT, 300.0, f['source_lon'], f['source_lat'], [0.0],
        )


def test_nan_consistency(ridge):
    """
    NaNs must form a contiguous trailing block in each ray — once a position
    is NaN every subsequent position must also be NaN.  No valid positions
    may appear after the first NaN (no gaps in the ray path).
    """
    r        = ridge
    azimuths = np.arange(0, 360, 10, dtype=float)   # 36 rays; some hit boundaries

    ray_lon, ray_lat, ray_dir = tt.trace_rays(
        r['lon_arr'], r['lat_arr'], r['depth'],
        DT, 7_200.0, r['source_lon'], r['source_lat'], azimuths,
    )

    for name, arr in [('ray_lon', ray_lon), ('ray_lat', ray_lat), ('ray_dir', ray_dir)]:
        for i in range(arr.shape[0]):
            nan_mask = np.isnan(arr[i])
            if nan_mask.any():
                first_nan = int(np.argmax(nan_mask))
                assert nan_mask[first_nan:].all(), (
                    f"{name} ray {i} (az={azimuths[i]}°): "
                    f"valid value found after first NaN at step {first_nan}"
                )


def test_azimuthal_symmetry(ridge):
    """
    Rays at azimuth θ and (360 − θ) are meridional mirror images: they travel
    the same distance northward but diverge east vs west by equal amounts.
    For an equatorial source on a bathymetry symmetric about the equator this
    requires:
      - longitude deviations from source are equal and opposite: Δlon(θ) = −Δlon(360−θ)
      - latitude traces are identical: lat(θ) = lat(360−θ)
    """
    r        = ridge
    azimuths = np.array([30.0, 330.0])   # NNE and NNW — both stay within bounds

    ray_lon, ray_lat, _ = tt.trace_rays(
        r['lon_arr'], r['lat_arr'], r['depth'],
        DT, 3_600.0, r['source_lon'], r['source_lat'], azimuths,
    )

    assert not np.isnan(ray_lon[0]).any(), "Ray at 30° terminated early — extend grid or shorten max_time"
    assert not np.isnan(ray_lon[1]).any(), "Ray at 330° terminated early — extend grid or shorten max_time"

    # Longitude deviations from the source are mirror-symmetric:
    # ray at 30° drifts east by δ; ray at 330° drifts west by the same δ.
    # Tolerance is 1e-5° (~1 m) to allow for floating-point asymmetry in the
    # RK4 grid-cell index lookup accumulating over the integration.
    src_lon = r['source_lon']
    np.testing.assert_allclose(
        ray_lon[0] - src_lon, -(ray_lon[1] - src_lon), atol=1e-5,
        err_msg="Longitude deviations are not mirror-symmetric about the source longitude",
    )
    # Both rays head equally northward, so latitude traces must be identical.
    np.testing.assert_allclose(ray_lat[0], ray_lat[1], atol=1e-5,
                               err_msg="Latitude traces are not equal for meridional mirror rays")


def test_ridge_slows_westward_ray(ridge):
    """
    A westward ray crossing the submarine ridge must travel less distance by
    max_time than the same ray on a uniformly deep ocean.  This confirms that
    the slowness field correctly reduces wave speed in shallow water (Snell's law).
    """
    r        = ridge
    MAX_TIME = 7_200.0

    # Ridge bathymetry
    ray_lon_ridge, ray_lat_ridge, _ = tt.trace_rays(
        r['lon_arr'], r['lat_arr'], r['depth'],
        DT, MAX_TIME, r['source_lon'], r['source_lat'], np.array([270.0]),
    )

    # Flat deep-ocean reference at the same background depth
    flat_depth = np.full_like(r['depth'], r['deep_ocean'])
    ray_lon_flat, ray_lat_flat, _ = tt.trace_rays(
        r['lon_arr'], r['lat_arr'], flat_depth,
        DT, MAX_TIME, r['source_lon'], r['source_lat'], np.array([270.0]),
    )

    assert not np.isnan(ray_lon_ridge[0]).any(), "Ridge ray terminated early"
    assert not np.isnan(ray_lon_flat[0]).any(),  "Flat-ocean ray terminated early"

    def arc_length_m(lon_deg, lat_deg, src_lon, src_lat):
        """Spherical arc length from source to each recorded position, in metres."""
        theta_0 = (90.0 - src_lat) * DEG_TO_RAD
        phi_0   = src_lon * DEG_TO_RAD
        theta   = (90.0 - lat_deg) * DEG_TO_RAD
        phi     = lon_deg * DEG_TO_RAD
        cos_arc = np.clip(
            np.cos(theta) * np.cos(theta_0)
            + np.sin(theta) * np.sin(theta_0) * np.cos(phi - phi_0),
            -1.0, 1.0,
        )
        return np.arccos(cos_arc) * EARTH_RADIUS

    arc_ridge = arc_length_m(
        ray_lon_ridge[0, -1], ray_lat_ridge[0, -1],
        r['source_lon'], r['source_lat'],
    )
    arc_flat = arc_length_m(
        ray_lon_flat[0, -1], ray_lat_flat[0, -1],
        r['source_lon'], r['source_lat'],
    )

    assert arc_ridge < arc_flat, (
        f"Ridge ray ({arc_ridge/1e3:.1f} km) should be shorter than "
        f"flat-ocean ray ({arc_flat/1e3:.1f} km) at t={MAX_TIME:.0f} s"
    )
