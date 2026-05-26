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

TsunamiTrace computes these ray paths by integrating the spherical ray-tracing equations (Snell's law adapted for a sphere) from one or more sources across a fan of azimuths. A single point source returns a `(n_azimuths, n_steps)` array; an array of sources (finite-fault sub-faults, for example) returns `(n_sources, n_azimuths, n_steps)` — all sources are integrated in a single vectorised RK4 pass. The state vector at each time step is (colatitude θ, longitude φ, ray direction ψ), governed by:

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
│   ├── __init__.py        # Public API: trace_rays(), load_bathymetry(), grid_travel_times(), grid_azimuths()
│   ├── raytracing.py      # trace_rays(): builds slowness field, fans rays from one or more sources
│   ├── _rungekutta.py     # _integrate_rays(): vectorised RK4 integrator for all rays
│   ├── io.py              # load_bathymetry(): bathymetry file loaders
│   └── analysis.py        # grid_travel_times(), grid_azimuths(): post-processing of ray output
├── data/
│   ├── cascadia.xyz            # SRTM30+ 30 arc-second bathymetry, Cascadia (Git LFS)
│   ├── NE_pacific_4arcmin.nc  # ETOPO2 4 arc-minute bathymetry, NE Pacific / Alaska
│   ├── CSZ_trench.txt          # CSZ trench axis polyline (plotting reference)
│   ├── CSZ_max_def.txt         # CSZ max-deformation path (source line for notebook 05)
│   ├── CSZ_US_coast.txt        # Pacific coast receiver points, N. California → Washington
│   ├── CSZ_CA_coast.txt        # Pacific coast receiver points, British Columbia
│   └── 1964_slip_region.txt    # 1964 Alaska earthquake rupture polygon (notebook 03)
├── examples/
│   ├── 01_ridge_refraction.ipynb              # Ray refraction across a synthetic submarine ridge
│   ├── 02_cascadia_travel_times.ipynb         # Regional travel times for a Cascadia megathrust scenario
│   ├── 03_alaska_point_vs_finite_fault.ipynb  # Point source vs finite fault, 1964 Alaska earthquake
│   ├── 04_DART_arrival_times.ipynb            # Predicted DART arrival times, 2021 Chignik M8.2
│   └── 05_CSZ_coastal_arrival_times.ipynb    # Minimum tsunami arrival times at the US/BC coast, CSZ
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

# Single point source — returns shape (n_azimuths, n_steps)
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
# ray_lon, ray_lat, ray_dir: shape (n_azimuths, n_steps), NaN-padded after boundary exit

# Finite fault / multi-source — returns shape (n_sources, n_azimuths, n_steps)
src_lons = np.array([155.0, 157.0, 159.0])   # sub-fault centroids
src_lats = np.array([-1.0,   0.0,   1.0])

ray_lon, ray_lat, ray_dir = tt.trace_rays(
    lon_arr, lat_arr, depth,
    dt=30, max_time=7200,
    source_lon=src_lons,
    source_lat=src_lats,
    azimuths_deg=np.arange(0, 360, 2),
)
# All sources and azimuths are integrated in a single vectorised RK4 pass.
# Pass the (n_sources, n_azimuths, n_steps) arrays directly to grid_travel_times —
# it flattens them internally and keeps the minimum travel time per cell across
# all sources, giving a true first-arrival map for the entire finite fault.
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

# Pass fill=False to see raw ray coverage — bins with no ray hit are NaN
# and can be rendered in grey to diagnose shadow zones:
_, _, travel_time_raw = tt.grid_travel_times(
    ray_lon, ray_lat, dt=30,
    lon_arr=lon_arr, lat_arr=lat_arr, depth=depth,
    bin_deg=0.1, fill=False,
)
```

### Source azimuth map

```python
# Great-circle bearing from the source to every bin centre.
# Useful for identifying which azimuths have poor ray coverage
# and need a denser ray fan.
azimuth = tt.grid_azimuths(source_lon, source_lat, lon_bin, lat_bin)
# azimuth: shape (n_lat_bin, n_lon_bin), degrees clockwise from north, range [0, 360)

# twilight is a circular colormap — 0° and 360° match seamlessly
plt.pcolormesh(lon_bin, lat_bin, azimuth, cmap='twilight', vmin=0, vmax=360,
               shading='nearest')
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

The test suite has 16 tests across two files:

| File | Tests |
|------|-------|
| `tests/test_rk4.py` | RK4 integrator unit tests: great-circle accuracy across 5 azimuths, zero longitude drift for due-north ray, zero latitude drift for due-east equatorial ray, arc-length error < 1 m over 1 hour |
| `tests/test_trace_rays.py` | Integration tests: output shape (scalar and array sources), invalid-depth-shape error, NaN consistency, meridional symmetry, Snell's law slowdown across a submarine ridge, Snell's law refraction direction on a north-deepening gradient, multi-source output shape, multi-source results match individual single-source calls |

## Examples

`examples/01_ridge_refraction.ipynb`: Jupyter notebook demonstrating ray refraction across a synthetic N-S submarine ridge. A Gaussian ridge sits between the source and the western edge of the domain; the shallow ridge crest slows the wave from ~221 m/s (deep ocean) to ~63 m/s (ridge crest), bending rays toward the normal to the isobaths.

`examples/02_cascadia_travel_times.ipynb`: Real-bathymetry example using an SRTM30+ 30 arc-second grid of the Cascadia subduction zone (offshore Washington / Oregon / British Columbia). Traces 360,000 rays from a source on the locked zone of the megathrust (47.65°N, 125.50°W) and produces several diagnostic and final maps: a filled first-arrival travel time map (`fill=True`), a raw ray-coverage map with NaN cells rendered in grey (`fill=False`), a combined travel-time-plus-rays overlay, and an azimuth map (`tt.grid_azimuths`) with the travel-time contours to identify which source azimuths have sparse ray coverage. Requires `scipy` (`pip install -e ".[examples]"`).

`examples/03_alaska_point_vs_finite_fault.ipynb`: Trans-oceanic travel time example using an ETOPO2 4 arc-minute NetCDF grid of the NE Pacific. Models the 1964 Alaska earthquake source (nudged offshore to 60.07°N, 146.68°W) and integrates 360,000 rays for 12 hours to capture trans-oceanic propagation. Compares point-source and finite-fault approaches. Requires `scipy` and `netCDF4` (`pip install -e ".[examples]"`).

`examples/04_DART_arrival_times.ipynb`: Regional travel time example for the 2021 Chignik M8.2 earthquake (55.36°N, 157.89°W) in the Gulf of Alaska. Traces 360 rays for 3 hours and produces a first-arrival travel time map for the Gulf of Alaska, then samples predicted arrival times at three nearby DART buoys (46414, 46409, 46410) and annotates them on the map. Requires `scipy` and `netCDF4` (`pip install -e ".[examples]"`).

`examples/05_CSZ_coastal_arrival_times.ipynb`: Near-field CSZ scenario addressing the question "how many minutes after the shaking does the first wave arrive?". Distributes 150 sources along the `CSZ_max_def` path (the zone of maximum seafloor deformation, between the trench and coast), traces 54,000 rays simultaneously, and grids the minimum first-arrival time across all sources. Samples the result at 200 US coast points (northern California → Washington) and 150 Canadian coast points (British Columbia) and displays them as a scatter plot coloured by arrival time overlaid on the regional travel time map. Requires `scipy` (`pip install -e ".[examples]"`).

`data/cascadia.xyz` is stored in Git LFS (51 MB). After cloning, run `git lfs pull` if it is not automatically retrieved. All other data files (`NE_pacific_4arcmin.nc`, `CSZ_*.txt`, `1964_slip_region.txt`) are small enough to be committed directly.

To run the examples:

```bash
conda activate tsunamitrace
jupyter notebook examples/01_ridge_refraction.ipynb
jupyter notebook examples/02_cascadia_travel_times.ipynb
jupyter notebook examples/03_alaska_point_vs_finite_fault.ipynb
jupyter notebook examples/04_DART_arrival_times.ipynb
jupyter notebook examples/05_CSZ_coastal_arrival_times.ipynb
```

## Performance

All rays are integrated simultaneously using NumPy vectorisation: the state of every ray is held in arrays of shape `(n_rays,)` and a single RK4 pass advances all rays at once using array operations. A boolean `alive` mask suppresses updates for rays that have exited the grid, hit shallow water, or reached a dry cell, so no work is wasted on terminated rays.

On a 350 × 400 grid with a 30 s time step and 4-hour integration this gives roughly:

| Ray count | Wall time |
|-----------|-----------|
| 180 rays  | ~70 ms    |
| 720 rays  | ~125 ms   |

The Cascadia example runs 360,000 rays on a 1080 × 1560 grid (dt = 180 s, 2.5-hour integration) in roughly a minute on a laptop.

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
