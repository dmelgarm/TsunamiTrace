# TsunamiTrace

Python package for tsunami ray tracing on a sphere using 4th-order Runge-Kutta integration.

## What it does

Tsunami waves travel at a speed determined by the local water depth: `c = sqrt(g * depth)`. Because depth varies across the ocean floor, rays continuously refract — bending toward shallower regions where they travel more slowly. This is the ocean-surface analogue of seismic ray theory.

TsunamiTrace computes these ray paths by integrating the spherical ray-tracing equations (Snell's law adapted for a sphere) from a single point source across a fan of azimuths. The state vector at each time step is (colatitude θ, longitude φ, ray direction ψ), governed by:

```
dθ/dt  =  cos(ψ) / (n · R)
dφ/dt  =  sin(ψ) / (n · R · sin(θ))
dψ/dt  = −sin(ψ) · ∂n/∂θ / (n² · R)
         + cos(ψ) · ∂n/∂φ / (n² · R · sin(θ))
         − sin(ψ) · cos(θ) / (n · R · sin(θ))
```

where `n = 1/c` is the slowness and `R` is Earth's radius. The third term in the direction equation is a spherical correction for meridian convergence. Each ray is integrated until it reaches the grid boundary, water shallower than 10 m, or a dry cell.

## Attribution

This implementation follows the ray-tracing methodology described in:

> Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017). Fault slip distribution of the 2016 Fukushima earthquake estimated from tsunami waveforms. *Pure and Applied Geophysics*, 174(8), 2925–2943. https://doi.org/10.1007/s00024-017-1590-2

The original MATLAB implementation by Aditya Gusman is available at:
https://github.com/adityagusman/tsunami-raytracing

## Package structure

```
TsunamiTrace/
├── TsunamiTrace/
│   ├── __init__.py        # Public API — exposes trace_rays()
│   ├── raytracing.py      # trace_rays(): builds slowness field, fans rays
│   └── _rungekutta.py     # _integrate_rays(): vectorised RK4 integrator for all rays
├── examples/
│   └── ridge_refraction.ipynb   # Ray refraction across a synthetic submarine ridge
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

```python
import numpy as np
import TsunamiTrace as tt

ray_lon, ray_lat, ray_dir = tt.trace_rays(
    lon_arr,      # 1-D array of longitudes (degrees)
    lat_arr,      # 1-D array of latitudes (degrees)
    depth,        # 2-D depth array, shape (n_lon, n_lat), metres
    dt=30,        # time step (seconds)
    max_time=7200,
    source_lon=157.0,
    source_lat=0.0,
    azimuths_deg=np.arange(0, 360, 2),
)
# ray_lon, ray_lat, ray_dir — shape (n_rays, n_steps), NaN-padded after boundary exit
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

The test suite has 13 tests across two files:

| File | Tests |
|------|-------|
| `tests/test_rk4.py` | RK4 integrator unit tests: great-circle accuracy across 5 azimuths, zero longitude drift for due-north ray, zero latitude drift for due-east equatorial ray, arc-length error < 1 m over 1 hour |
| `tests/test_trace_rays.py` | Integration tests: output shape, invalid-depth-shape error, NaN consistency, meridional symmetry, Snell's law slowdown across a submarine ridge |

## Examples

`examples/ridge_refraction.ipynb` — Jupyter notebook demonstrating ray refraction across a synthetic N–S submarine ridge. A Gaussian ridge sits between the source and the western edge of the domain; the shallow ridge crest slows the wave from ~221 m/s (deep ocean) to ~63 m/s (ridge crest), bending rays toward the normal to the isobaths.

To run it:

```bash
conda activate tsunamitrace
jupyter notebook examples/ridge_refraction.ipynb
```

## Performance

All rays are integrated simultaneously using NumPy vectorisation: the state of every ray is held in arrays of shape `(n_rays,)` and a single RK4 pass advances all rays at once using array operations. A boolean `alive` mask suppresses updates for rays that have exited the grid, hit shallow water, or reached a dry cell, so no work is wasted on terminated rays.

On a 350 × 400 grid with a 30 s time step and 4-hour integration this gives roughly:

| Ray count | Wall time |
|-----------|-----------|
| 180 rays  | ~70 ms    |
| 720 rays  | ~125 ms   |

Scaling with ray count is much flatter than a sequential loop because the per-ray overhead is absorbed into NumPy's C-level inner loop.

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib
- pytest (for tests, optional)
