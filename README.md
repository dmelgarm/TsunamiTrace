# TsunamiTrace

Python package for tsunami ray tracing on a sphere using 4th-order Runge-Kutta integration.

## Applications

The most common use of TsunamiTrace is computing **first-arrival travel time maps**: grids that show how many minutes or hours after a geophysical source (earthquake, volcanic eruption, or submarine landslide) a tsunami would reach any point in the ocean.  These maps underpin two broad classes of work:

**Hazard and emergency response**
- Estimating evacuation lead times for coastal communities
- Prioritising warning dissemination by arrival order
- Rapid post-event impact assessment when source location is known

**Observational geophysics**
- Windowing DART buoy and tide-gauge records: knowing the predicted arrival time lets you extract the tsunami signal from the background noise and cut analysis windows of the right length
- Cross-checking hydrodynamic model runs against expected wave front timing
- Back-projection of observed arrivals to constrain rupture location and extent

## What it does

Tsunami waves travel at a speed determined by the local water depth: `c = sqrt(g * depth)`. Because depth varies across the ocean floor, rays continuously refract, bending toward shallower regions where they travel more slowly. This is the ocean-surface analogue of seismic ray theory.

TsunamiTrace computes these ray paths by integrating the spherical ray-tracing equations (Snell's law adapted for a sphere) from a single point source across a fan of azimuths. The state vector at each time step is (colatitude θ, longitude φ, ray direction ψ), governed by:

```
dθ/dt  =  cos(ψ) / (n · R)
dφ/dt  =  sin(ψ) / (n · R · sin(θ))
dψ/dt  = −sin(ψ) · ∂n/∂θ / (n² · R)
         + cos(ψ) · ∂n/∂φ / (n² · R · sin(θ))
         − sin(ψ) · cos(θ) / (n · R · sin(θ))
```

where `n = 1/c` is the slowness and `R` is Earth's radius. The third term in the direction equation is a spherical correction for meridian convergence. Each ray is integrated until it reaches the grid boundary, water shallower than 10 m, or a dry cell.

The colatitude step `dcolat_rad` is **signed**: when `lat_arr` is ascending (the usual convention), colatitude decreases as the array index increases, so `dcolat_rad` is negative. The sign governs grid-cell index snapping and the direction of the slowness gradient; reversing it mirrors the effective latitude axis and sends rays in the wrong direction.

## Attribution

This implementation follows the ray-tracing methodology described in:

> Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017). Fault slip distribution of the 2016 Fukushima earthquake estimated from tsunami waveforms. *Pure and Applied Geophysics*, 174(8), 2925–2943. https://doi.org/10.1007/s00024-017-1590-2

The original MATLAB implementation by Aditya Gusman is available at:
https://github.com/adityagusman/tsunami-raytracing

## Package structure

```
TsunamiTrace/
├── TsunamiTrace/
│   ├── __init__.py        # Public API: trace_rays(), load_bathymetry(), grid_travel_times()
│   ├── raytracing.py      # trace_rays(): builds slowness field, fans rays
│   ├── _rungekutta.py     # _integrate_rays(): vectorised RK4 integrator for all rays
│   ├── io.py              # load_bathymetry(): bathymetry file loaders
│   └── analysis.py        # grid_travel_times(): post-processing of ray output
├── data/
│   ├── cascadia.xyz            # GEBCO 30 arc-second bathymetry, Cascadia (Git LFS)
│   └── NE_pacific_4arcmin.nc  # GEBCO 4 arc-minute bathymetry, NE Pacific / Alaska
├── examples/
│   ├── ridge_refraction.ipynb     # Ray refraction across a synthetic submarine ridge
│   └── cascadia_travel_times.ipynb  # Regional travel times for a Cascadia megathrust scenario
├── tests/
│   ├── test_rk4.py        # Unit tests for the RK4 integrator (great-circle accuracy)
│   └── test_trace_rays.py # Integration tests for trace_rays() (shape, symmetry, Snell's law)
├── pyproject.toml
└── README.md
```

## Installation

Requires Python 3.9+ and a working conda or pip environment.

```bash
# Clone the repository
git clone https://github.com/dmelgarm/TsunamiTrace.git
cd TsunamiTrace

# Install in editable mode (recommended for development)
pip install -e .

# Or install with test dependencies
pip install -e ".[dev]"
```

### Conda environment (recommended)

```bash
conda create -n tsunamitrace python=3.11
conda activate tsunamitrace
pip install -e ".[dev]"
```

## Usage

### Loading bathymetry

The format is detected automatically from the file extension.

```python
import TsunamiTrace as tt

# Three-column ASCII (.xyz, .txt, …)
lon_arr, lat_arr, depth = tt.load_bathymetry('data/mybathy.xyz')

# NetCDF (.nc, .nc4) — depth variable auto-detected from common names
# (z, elevation, topo, depth, Band1, …)
lon_arr, lat_arr, depth = tt.load_bathymetry('data/mybathy.nc')

# NetCDF with an unusual variable name — specify it explicitly
lon_arr, lat_arr, depth = tt.load_bathymetry('data/mybathy.nc', depth_var='sea_floor_depth')

# negate=True (default) converts from the standard geographic convention
# (ocean negative) to the TsunamiTrace convention (ocean positive).
# Pass negate=False if your file already stores ocean depth as positive.
lon_arr, lat_arr, depth = tt.load_bathymetry('data/mybathy.nc', negate=False)

# depth shape is (n_lon, n_lat), ready for trace_rays.
# For matplotlib contour plots transpose to row-major (n_lat, n_lon):
#   plt.contourf(lon_arr, lat_arr, depth.T)
```

### Ray tracing

```python
import numpy as np
import TsunamiTrace as tt

lon_arr, lat_arr, depth = tt.load_bathymetry('data/mybathy.xyz')

ray_lon, ray_lat, ray_dir = tt.trace_rays(
    lon_arr,      # 1-D array of longitudes (degrees)
    lat_arr,      # 1-D array of latitudes (degrees)
    depth,        # 2-D depth array, shape (n_lon, n_lat), positive = ocean
    dt=30,        # time step (seconds)
    max_time=7200,
    source_lon=157.0,
    source_lat=0.0,
    azimuths_deg=np.arange(0, 360, 2),
)
# ray_lon, ray_lat, ray_dir: shape (n_rays, n_steps), NaN-padded after boundary exit
```

### Travel time map

```python
# Grid the ray output onto a regular lon/lat grid, keeping the first-arrival
# time in each cell.  bin_deg controls the output resolution independently
# of the bathymetry grid spacing.
lon_bin, lat_bin, travel_time = tt.grid_travel_times(
    ray_lon, ray_lat,
    dt=30,                  # must match the dt used in trace_rays
    lon_arr=lon_arr,
    lat_arr=lat_arr,
    depth=depth,
    bin_deg=0.1,            # output cell size in degrees
    fill=True,              # interpolate over empty shadow-zone bins
)
# travel_time: shape (n_lat_bin, n_lon_bin), hours, NaN over land
# Ready for matplotlib:
import matplotlib.pyplot as plt
plt.contourf(lon_bin, lat_bin, travel_time, cmap='plasma_r')
```

## Running the tests

```bash
cd TsunamiTrace
pytest
```

Or with verbose output:

```bash
pytest -v
```

The test suite has 14 tests across two files:

| File | Tests |
|------|-------|
| `tests/test_rk4.py` | RK4 integrator unit tests: great-circle accuracy across 5 azimuths, zero longitude drift for due-north ray, zero latitude drift for due-east equatorial ray, arc-length error < 1 m over 1 hour |
| `tests/test_trace_rays.py` | Integration tests: output shape, invalid-depth-shape error, NaN consistency, meridional symmetry, Snell's law slowdown across a submarine ridge, Snell's law refraction direction on a north-deepening gradient |

## Examples

`examples/ridge_refraction.ipynb`: Jupyter notebook demonstrating ray refraction across a synthetic N-S submarine ridge. A Gaussian ridge sits between the source and the western edge of the domain; the shallow ridge crest slows the wave from ~221 m/s (deep ocean) to ~63 m/s (ridge crest), bending rays toward the normal to the isobaths.

`examples/cascadia_travel_times.ipynb`: Real-bathymetry example using a GEBCO-derived 30 arc-second grid of the Cascadia subduction zone (offshore Washington / Oregon / British Columbia). Traces 36,000 rays from a source on the locked zone of the megathrust (47.86°N, 124.91°W) and produces a first-arrival travel time map for the Pacific Northwest coast using `tt.grid_travel_times`. Requires `scipy` (`pip install -e ".[examples]"`).

`data/cascadia.xyz` is stored in Git LFS (51 MB). After cloning, run `git lfs pull` if it is not automatically retrieved. `data/NE_pacific_4arcmin.nc` is small enough (2.9 MB) to be committed directly.

To run the examples:

```bash
conda activate tsunamitrace
jupyter notebook examples/ridge_refraction.ipynb
jupyter notebook examples/cascadia_travel_times.ipynb
```

## Performance

All rays are integrated simultaneously using NumPy vectorisation: the state of every ray is held in arrays of shape `(n_rays,)` and a single RK4 pass advances all rays at once using array operations. A boolean `alive` mask suppresses updates for rays that have exited the grid, hit shallow water, or reached a dry cell, so no work is wasted on terminated rays.

On a 350 × 400 grid with a 30 s time step and 4-hour integration this gives roughly:

| Ray count | Wall time |
|-----------|-----------|
| 180 rays  | ~70 ms    |
| 720 rays  | ~125 ms   |

The Cascadia example runs 1800 rays on a 1080 × 1560 grid (dt = 60 s, 4-hour integration) in a few seconds on a laptop.

Scaling with ray count is much flatter than a sequential loop because the per-ray overhead is absorbed into NumPy's C-level inner loop.

## Dependencies

| Package | Required for |
|---------|-------------|
| Python 3.9+ | core requirement |
| NumPy | core ray tracing |
| Matplotlib | core requirement |
| pandas | `load_bathymetry()` ASCII path (optional; falls back to `numpy.loadtxt` if absent) |
| netCDF4 | `load_bathymetry()` NetCDF path (`pip install -e ".[examples]"`) |
| scipy | `grid_travel_times()` and examples |
| pytest | tests (`pip install -e ".[dev]"`) |
