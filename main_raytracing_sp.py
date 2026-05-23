"""
Synthetic-bathymetry test driver for tsunami ray tracing.

Builds a smooth test bathymetry from the peaks() function, runs the ray-tracer
across a full azimuthal fan from a single point source, and saves a plot of all
ray paths overlaid on the bathymetry contours.

Implements the ray-tracing approach from:
  Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
  Fault slip distribution of the 2016 Fukushima earthquake estimated from
  tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.

Originally implemented in MATLAB; this is the Python port.

Run:
    python main_raytracing_sp.py

Output:
    raytracing_sp_c.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from raytracing import raytracing_sp


# ── peaks bathymetry ──────────────────────────────────────────────────────────

def make_peaks_bathymetry(n_grid=200, depth_scale_m=1000.0):
    """
    Build a smooth synthetic bathymetry using the peaks() function.

    The peaks formula produces a mix of hills and valleys suitable for testing
    ray refraction across variable depth. The grid is indexed [lon_idx, lat_idx]
    (first axis = longitude) to match the convention expected by raytracing_sp.

    Parameters
    ----------
    n_grid : int
        Grid resolution.
    depth_scale_m : float
        Multiplier on the raw peaks values to convert to metres.

    Returns
    -------
    lon_arr : ndarray, shape (n_grid,)
        Longitude coordinates — linspace(-3, 3, n_grid).
    lat_arr : ndarray, shape (n_grid,)
        Latitude coordinates — linspace(-3, 3, n_grid).
    depth : ndarray, shape (n_grid, n_grid)
        Bathymetry in metres, indexed [lon_idx, lat_idx].
        Positive = ocean, negative = land.
    """
    t = np.linspace(-3.0, 3.0, n_grid)
    X, Y = np.meshgrid(t, t)

    # Standard peaks formula scaled to metres
    Z = (
        3.0 * (1 - X)**2 * np.exp(-X**2 - (Y + 1)**2)
        - 10.0 * (X / 5 - X**3 - Y**5) * np.exp(-X**2 - Y**2)
        - (1.0 / 3) * np.exp(-(X + 1)**2 - Y**2)
    ) * depth_scale_m

    depth = Z   # indexed [lon_idx, lat_idx] as raytracing_sp expects

    return t, t, depth   # lon_arr, lat_arr, depth


# ── simulation parameters ─────────────────────────────────────────────────────
DT           = 10            # integration time step, seconds
MAX_TIME     = 7_200         # maximum integration time, seconds  (2 hours)
AZIMUTHS_DEG = np.arange(0, 361, 1, dtype=float)   # 0°…360° in 1° steps
SOURCE_LON   = 1.5           # source longitude (dimensionless grid units)
SOURCE_LAT   = 0.0           # source latitude  (dimensionless grid units)

# ── build bathymetry ──────────────────────────────────────────────────────────
lon_arr, lat_arr, depth = make_peaks_bathymetry(n_grid=200, depth_scale_m=1000.0)

source_depth = depth[
    int(np.argmin(np.abs(lon_arr - SOURCE_LON))),
    int(np.argmin(np.abs(lat_arr - SOURCE_LAT))),
]
print(f"Grid        : {len(lon_arr)}×{len(lat_arr)} cells, "
      f"lon [{lon_arr[0]:.1f}, {lon_arr[-1]:.1f}], "
      f"lat [{lat_arr[0]:.1f}, {lat_arr[-1]:.1f}]")
print(f"Depth range : {depth.min():.0f} m  to  {depth.max():.0f} m")
print(f"Source      : lon={SOURCE_LON}, lat={SOURCE_LAT}, "
      f"depth at source ≈ {source_depth:.0f} m")

# ── ray tracing ───────────────────────────────────────────────────────────────
print(f"\nTracing {len(AZIMUTHS_DEG)} rays "
      f"(dt={DT} s, max_time={MAX_TIME} s) …")

ray_lon, ray_lat, ray_dir = raytracing_sp(
    lon_arr, lat_arr, depth,
    DT, MAX_TIME,
    SOURCE_LON, SOURCE_LAT,
    AZIMUTHS_DEG,
)

n_rays, n_steps = ray_lon.shape
n_terminated = int(np.sum(np.isnan(ray_lon[:, -1])))
print(f"Done.  Output shape: {ray_lon.shape}")
print(f"Rays terminated before max_time: {n_terminated}/{n_rays}")

# ── figure ────────────────────────────────────────────────────────────────────
# depth.T is used because contourf expects Z[lat_idx, lon_idx] (row = lat),
# but depth is indexed [lon_idx, lat_idx].

fig, ax = plt.subplots(figsize=(7, 7))

levels = np.linspace(depth.min(), depth.max(), 25)
cf = ax.contourf(lon_arr, lat_arr, depth,
                 levels=levels, cmap='RdBu_r', alpha=0.75)

# Zero-depth contour marks the coastline
ax.contour(lon_arr, lat_arr, depth,
           levels=[0], colors='b', linewidths=1.2)


ax.contour(lon_arr, lat_arr, depth,
           levels=levels, colors='k', linewidths=0.3, alpha=0.4)



# NaN values in the ray arrays break the plotted line at termination points
for ray_idx in range(n_rays):
    ax.plot(ray_lon[ray_idx], ray_lat[ray_idx],
            color='k', linewidth=0.4, alpha=0.55)

# Source marker — 5-pointed star
ax.plot(SOURCE_LON, SOURCE_LAT,
        marker=(5, 1),
        markersize=14,
        markerfacecolor='r',
        markeredgecolor='k',
        markeredgewidth=0.8,
        linestyle='none',
        zorder=5)

ax.set_aspect('equal')
ax.set_xlabel('Longitude (grid units)')
ax.set_ylabel('Latitude (grid units)')
ax.set_title('Tsunami ray tracing — synthetic peaks() bathymetry\n'
             f'source ({SOURCE_LON}, {SOURCE_LAT}), '
             f'{len(AZIMUTHS_DEG)} rays, dt={DT} s')

cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Depth (m)')
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{x:,.0f}'
))

plt.tight_layout()
plt.savefig('raytracing_sp_c.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nSaved: raytracing_sp_c.png")
