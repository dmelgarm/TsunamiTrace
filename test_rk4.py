"""
Validate rt_rungekutta_sp on a flat (constant-depth) ocean.

On a uniform-depth ocean the slowness gradients are zero everywhere, so each
ray must follow a great-circle path at a constant speed of sqrt(g*d).  Two
analytically exact cases anchor the numerical checks:

  azimuth = 0°  (due north)
      dphi/dt = 0  →  longitude is constant for the entire integration.

  azimuth = 90° (due east, source at equator)
      The spherical correction term vanishes at theta = pi/2, so dtheta/dt = 0
      →  latitude is constant for the entire integration.

Azimuths follow the standard geographic convention: 0=N, 90=E, 180=S, 270=W
(clockwise from north), matching the interface of rt_rungekutta_sp.

For all azimuths the integrated path is compared against the analytical
great-circle position computed from spherical trigonometry.

Tests the implementation of the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.
"""
import numpy as np
import matplotlib.pyplot as plt
from rt_rungekutta_sp import rt_rungekutta_sp

# ── physical and numerical constants ─────────────────────────────────────────
DEG_TO_RAD   = np.pi / 180.0
EARTH_RADIUS = 6_371_000.0   # m
G            = 9.8           # m/s²
DEPTH        = 4_000.0       # m  (flat ocean)

WAVE_SPEED = np.sqrt(G * DEPTH)        # ~197.99 m/s
SLOWNESS   = 1.0 / WAVE_SPEED         # s/m

# ── grid ─────────────────────────────────────────────────────────────────────
# 20° × 20° plate-carrée grid centred on the equator/prime-meridian.
# Spacing is uniform so dphi_rad == dcolat_rad.
GRID_SPACING_DEG = 0.1
lon_arr = np.arange(-10.0, 10.0 + GRID_SPACING_DEG, GRID_SPACING_DEG)
lat_arr = np.arange(-10.0, 10.0 + GRID_SPACING_DEG, GRID_SPACING_DEG)
n_lon = len(lon_arr)
n_lat = len(lat_arr)

depth_grid    = np.full((n_lon, n_lat), DEPTH)
slowness_grid = np.full((n_lon, n_lat), SLOWNESS)

# Flat ocean → all slowness gradients are exactly zero.
# Shapes mirror raytracing_sp.py: (n_lon-1, n_lat) and (n_lon, n_lat-1).
slowness_grad_phi   = np.zeros((n_lon - 1, n_lat))
slowness_grad_colat = np.zeros((n_lon, n_lat - 1))

dphi_rad   = GRID_SPACING_DEG * DEG_TO_RAD
dcolat_rad = GRID_SPACING_DEG * DEG_TO_RAD

# ── time ─────────────────────────────────────────────────────────────────────
DT       = 10.0          # s
MAX_TIME = 3_600.0       # s  (1 hour — ray travels ~6.4° on Earth's surface)
time_arr = np.arange(0.0, MAX_TIME + DT, DT)

# ── source: dead centre of the grid ──────────────────────────────────────────
SOURCE_LON = 0.0
SOURCE_LAT = 0.0
source_ix  = int(np.argmin(np.abs(lon_arr - SOURCE_LON)))
source_iy  = int(np.argmin(np.abs(lat_arr - SOURCE_LAT)))


# ── analytical great-circle path ─────────────────────────────────────────────
def great_circle(source_lat_deg, source_lon_deg, azimuth_deg, t, wave_speed):
    """
    Return (lat_deg, lon_deg) arrays for a great-circle ray launched from
    (source_lat_deg, source_lon_deg) at azimuth_deg (measured from north in
    the spherical convention of the ODE system).

    Derivation uses the spherical law of cosines (side) for the colatitude
    and the four-parts formula / law of sines for the longitude increment.
    """
    theta_0 = (90.0 - source_lat_deg) * DEG_TO_RAD
    phi_0   = source_lon_deg * DEG_TO_RAD
    z_0     = azimuth_deg * DEG_TO_RAD
    alpha   = wave_speed * t / EARTH_RADIUS    # arc angle in radians

    # Standard geographic azimuth: 0=N, 90=E, 180=S (clockwise from north).
    # z_0=0 means the ray moves toward the north pole (decreasing colatitude).

    # Spherical law of cosines → colatitude along the great circle
    cos_theta = (np.cos(theta_0) * np.cos(alpha)
                 + np.sin(theta_0) * np.sin(alpha) * np.cos(z_0))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_gc  = np.arccos(cos_theta)

    # Longitude increment from the spherical law of sines + four-parts formula.
    # arctan2 resolves the hemisphere ambiguity that arcsin cannot.
    #   numerator:   sin(alpha) * sin(z_0)               [law of sines]
    #   denominator: cos(alpha)*sin(theta_0) - cos(theta_0)*sin(alpha)*cos(z_0)
    delta_phi = np.arctan2(
        np.sin(alpha) * np.sin(z_0),
        np.cos(alpha) * np.sin(theta_0) - np.cos(theta_0) * np.sin(alpha) * np.cos(z_0),
    )
    phi_gc = phi_0 + delta_phi

    return 90.0 - theta_gc / DEG_TO_RAD, phi_gc / DEG_TO_RAD  # lat, lon


# ── run integrator and compare ────────────────────────────────────────────────
azimuths = [0.0, 45.0, 90.0, 135.0, 180.0]

fig, (ax_path, ax_dev) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'RK4 integrator — flat {DEPTH:.0f} m ocean  '
             f'(wave speed = {WAVE_SPEED:.1f} m/s)')

print(f"Wave speed : {WAVE_SPEED:.4f} m/s")
print(f"Slowness   : {SLOWNESS:.6e} s/m")
print(f"Grid       : {n_lon}×{n_lat} cells, {GRID_SPACING_DEG}° spacing")
print(f"Source     : lon={SOURCE_LON}°, lat={SOURCE_LAT}°  "
      f"→  grid cell ({source_ix}, {source_iy})")
print()
print(f"{'Azimuth':>10}  {'Steps':>6}  "
      f"{'Max Δpos (°)':>14}  {'Max arc error (m)':>18}")
print("─" * 58)

all_max_devs = []

for azimuth in azimuths:
    phi_ray, theta_ray, ray_dir, ix_hist, iy_hist = rt_rungekutta_sp(
        time_arr, DT, dphi_rad, dcolat_rad,
        slowness_grid, slowness_grad_phi, slowness_grad_colat,
        azimuth, SOURCE_LON, SOURCE_LAT,
        source_ix, source_iy, n_lon, n_lat, depth_grid,
    )

    n_pts   = len(phi_ray)
    # phi_ray[k] is the state after k steps, i.e. at time k*DT.
    # This gives n_pts values including the initial condition, so the time
    # axis runs [0, DT, 2*DT, ..., (n_pts-1)*DT] — one element longer than
    # time_arr when the loop runs to completion.
    t_ray   = np.arange(n_pts) * DT
    lon_ray = phi_ray   / DEG_TO_RAD
    lat_ray = 90.0 - theta_ray / DEG_TO_RAD

    lat_gc, lon_gc = great_circle(SOURCE_LAT, SOURCE_LON, azimuth, t_ray, WAVE_SPEED)

    # ── deviation 1: angular distance from the expected great-circle point ───
    dev_deg = np.sqrt((lon_ray - lon_gc)**2 + (lat_ray - lat_gc)**2)

    # ── deviation 2: arc-distance error relative to wave_speed × t / R ──────
    theta_0_rad = (90.0 - SOURCE_LAT) * DEG_TO_RAD
    phi_0_rad   = SOURCE_LON * DEG_TO_RAD
    cos_arc = np.clip(
        np.cos(theta_ray) * np.cos(theta_0_rad)
        + np.sin(theta_ray) * np.sin(theta_0_rad) * np.cos(phi_ray - phi_0_rad),
        -1.0, 1.0,
    )
    arc_actual_m   = np.arccos(cos_arc) * EARTH_RADIUS
    arc_expected_m = WAVE_SPEED * t_ray
    max_arc_err_m  = float(np.abs(arc_actual_m - arc_expected_m).max())

    max_dev = float(dev_deg.max())
    all_max_devs.append(max_dev)

    print(f"{azimuth:>10.1f}°  {n_pts:>6d}  {max_dev:>14.4e}  {max_arc_err_m:>18.4f}")

    label = f'{azimuth:.0f}°'
    color = f'C{azimuths.index(azimuth)}'
    ax_path.plot(lon_ray, lat_ray, color=color, lw=2, label=label)
    ax_path.plot(lon_gc,  lat_gc,  color=color, lw=1, ls='--', alpha=0.5)
    ax_dev.semilogy(t_ray / 60.0, np.maximum(dev_deg, 1e-15), color=color, label=label)

# ── exact symmetry checks for the two analytically trivial azimuths ──────────
print()

# azimuth = 0°: northward ray — longitude must be exactly constant
phi_n, theta_n, *_ = rt_rungekutta_sp(
    time_arr, DT, dphi_rad, dcolat_rad,
    slowness_grid, slowness_grad_phi, slowness_grad_colat,
    0.0, SOURCE_LON, SOURCE_LAT,
    source_ix, source_iy, n_lon, n_lat, depth_grid,
)
lon_drift_north = float(np.abs(phi_n / DEG_TO_RAD - SOURCE_LON).max())
print(f"Due-north ray (az=0°)   max lon drift : {lon_drift_north:.2e}°  "
      f"(expected: 0)")

# azimuth = 90°: eastward ray at the equator — latitude must be exactly constant
_, theta_e, *_ = rt_rungekutta_sp(
    time_arr, DT, dphi_rad, dcolat_rad,
    slowness_grid, slowness_grad_phi, slowness_grad_colat,
    90.0, SOURCE_LON, SOURCE_LAT,
    source_ix, source_iy, n_lon, n_lat, depth_grid,
)
lat_drift_east = float(np.abs(90.0 - theta_e / DEG_TO_RAD - SOURCE_LAT).max())
print(f"Due-east  ray (az=90°)  max lat drift : {lat_drift_east:.2e}°  "
      f"(expected: 0)")

print()
print(f"Max deviation from straight line across all azimuths: "
      f"{max(all_max_devs):.4e}°")

# ── finalise plots ────────────────────────────────────────────────────────────
ax_path.plot(SOURCE_LON, SOURCE_LAT, 'k*', ms=10, zorder=5, label='source')
ax_path.set_xlabel('Longitude (°)')
ax_path.set_ylabel('Latitude (°)')
ax_path.set_title('Ray paths (solid) vs great circles (dashed grey)')
ax_path.legend(title='Azimuth', fontsize=8)
ax_path.set_aspect('equal')
ax_path.grid(True, alpha=0.3)

ax_dev.set_xlabel('Time (min)')
ax_dev.set_ylabel('Deviation from great circle (°)')
ax_dev.set_title('Positional error vs expected great-circle path')
ax_dev.legend(title='Azimuth', fontsize=8)
ax_dev.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('test_rk4_flat_ocean.png', dpi=150)
plt.show()
print("Plot saved: test_rk4_flat_ocean.png")
