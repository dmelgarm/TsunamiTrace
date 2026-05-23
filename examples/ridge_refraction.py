"""
ridge_refraction.py — synthetic ridge bathymetry example for TsunamiTrace.

Demonstrates ray refraction across a N-S submarine ridge sitting between the
source and the far side of the domain.  In deep water the wave speed is
c = sqrt(g * depth) ~ 220 m/s; over the ridge crest it drops to ~ 70 m/s.
Rays aimed at the ridge slow, compress, and deflect; rays that miss the ends
of the ridge travel almost unimpeded.

Run:
    python examples/ridge_refraction.py

Output:
    ridge_refraction.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import TsunamiTrace as tt

# ── grid ──────────────────────────────────────────────────────────────────────
N_LON, N_LAT = 350, 400
lon_arr = np.linspace(130.0, 165.0, N_LON)   # degrees east
lat_arr = np.linspace(-20.0,  20.0, N_LAT)   # degrees north

# LON/LAT have shape (n_lat, n_lon) — contourf row-major layout
LON, LAT = np.meshgrid(lon_arr, lat_arr)

# ── bathymetry ────────────────────────────────────────────────────────────────
# Deep-ocean background with a Gaussian N-S ridge.
# The ridge runs the full height of the domain (sigma_lat >> domain half-height)
# so rays cannot sneak around the ends — refraction across the ridge dominates.
DEEP_OCEAN  = 5_000.0   # m — background depth
RIDGE_SILL  =   400.0   # m — shallowest depth at the ridge crest
RIDGE_LON   =   147.0   # degrees east — ridge centre longitude
SIGMA_LON   =     1.5   # degrees — ridge half-width in longitude
SIGMA_LAT   =    30.0   # degrees — ridge half-length (longer than domain)

ridge_rise = (DEEP_OCEAN - RIDGE_SILL) * np.exp(
    - (LON - RIDGE_LON)**2 / (2 * SIGMA_LON**2)
    - LAT**2              / (2 * SIGMA_LAT**2)
)
Z = DEEP_OCEAN - ridge_rise     # (n_lat, n_lon) — plotting layout

# trace_rays expects depth[lon_idx, lat_idx]: transpose from (n_lat, n_lon)
depth = Z.T                     # (n_lon, n_lat)

# ── source ────────────────────────────────────────────────────────────────────
# Placed in the deep ocean east of the ridge.  Rays heading west cross the
# ridge; rays heading east travel unobstructed to the grid boundary.
SOURCE_LON = 157.0
SOURCE_LAT =   0.0

# ── integration parameters ────────────────────────────────────────────────────
# At deep-ocean speed (~221 m/s) the wavefront travels ~1,590 km in 2 hours,
# enough to cross the ridge (source-to-ridge ≈ 1,110 km at the equator).
DT           =  30       # s — time step
MAX_TIME     = 14_400    # s — 4 hours
AZIMUTHS_DEG = np.arange(0, 360, 2, dtype=float)   # every 2°, 180 rays

# ── ray tracing ───────────────────────────────────────────────────────────────
print(f"Source      : {SOURCE_LON}°E, {SOURCE_LAT}°N")
print(f"Ridge crest : {RIDGE_LON}°E, depth {RIDGE_SILL:.0f} m  "
      f"(wave speed {np.sqrt(9.8 * RIDGE_SILL):.1f} m/s)")
print(f"Deep ocean  : {DEEP_OCEAN:.0f} m  "
      f"(wave speed {np.sqrt(9.8 * DEEP_OCEAN):.1f} m/s)")
print(f"Tracing {len(AZIMUTHS_DEG)} rays (dt={DT} s, max {MAX_TIME//3600} h) …")

ray_lon, ray_lat, _ = tt.trace_rays(
    lon_arr, lat_arr, depth,
    DT, MAX_TIME,
    SOURCE_LON, SOURCE_LAT,
    AZIMUTHS_DEG,
)

n_done = int(np.sum(~np.isnan(ray_lon[:, -1])))
print(f"Done.  {n_done}/{len(AZIMUTHS_DEG)} rays reached max_time; "
      f"{len(AZIMUTHS_DEG) - n_done} terminated early (boundary or land).")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Filled depth contours
levels = np.linspace(RIDGE_SILL, DEEP_OCEAN, 30)
cf = ax.contourf(lon_arr, lat_arr, Z, levels=levels, cmap='Blues_r', alpha=0.85)

# Structural contours at round-number depths
depth_lines = [500, 1_000, 2_000, 3_000, 4_000]
cs = ax.contour(lon_arr, lat_arr, Z,
                levels=depth_lines, colors='steelblue',
                linewidths=0.7, alpha=0.6)
ax.clabel(cs, fmt='%d m', fontsize=7, inline=True)

# Ray paths — colour by azimuth so individual rays are distinguishable
cmap_rays = plt.cm.plasma
for i in range(len(AZIMUTHS_DEG)):
    c = cmap_rays(i / len(AZIMUTHS_DEG))
    ax.plot(ray_lon[i], ray_lat[i], color=c, linewidth=0.5, alpha=0.7)

# Source marker
ax.plot(SOURCE_LON, SOURCE_LAT,
        marker=(5, 1), markersize=14,
        markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=0.8,
        linestyle='none', zorder=6, label='source')

# Ridge crest line
ax.axvline(RIDGE_LON, color='w', linewidth=1.0, linestyle='--',
           alpha=0.6, label=f'ridge crest ({RIDGE_LON}°E)')

ax.set_xlim(lon_arr[0], lon_arr[-1])
ax.set_ylim(lat_arr[0], lat_arr[-1])
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title(
    f'Tsunami ray refraction — N–S submarine ridge\n'
    f'{len(AZIMUTHS_DEG)} rays, dt={DT} s, '
    f'max {MAX_TIME//3600} h, '
    f'ridge crest {RIDGE_SILL:.0f} m / deep ocean {DEEP_OCEAN:.0f} m'
)
ax.legend(loc='lower left', fontsize=8)
ax.set_aspect('equal')

cbar = fig.colorbar(cf, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Depth (m)')
cbar.ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
)

plt.tight_layout()
plt.savefig('ridge_refraction.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: ridge_refraction.png")
