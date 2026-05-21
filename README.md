# TsunamiTrace

Python implementation of tsunami ray tracing on a sphere using 4th-order Runge-Kutta integration.

## What it does

Tsunami waves travel at a speed determined by the local water depth: `c = sqrt(g * depth)`. Because depth varies across the ocean floor, rays continuously refract — bending toward shallower regions where they travel more slowly. This is the ocean-surface analogue of seismic ray theory.

TsunamiTrace computes these ray paths by integrating the spherical ray-tracing equations (Snell's law adapted for a sphere) from a single point source across a fan of azimuths. The state vector at each time step is (colatitude θ, longitude φ, ray direction ψ), governed by:

```
dθ/dt   =  cos(ψ) / (n · R)
dφ/dt   =  sin(ψ) / (n · R · sin(θ))
dψ/dt   = −sin(ψ) · ∂n/∂θ / (n² · R)
          + cos(ψ) · ∂n/∂φ / (n² · R · sin(θ))
          − sin(ψ) · cos(θ) / (n · R · sin(θ))
```

where `n = 1/c` is the slowness and `R` is Earth's radius. The third term in the direction equation is a spherical correction for meridian convergence.

Each ray is integrated until it reaches the grid boundary, water shallower than 10 m, or a dry cell.

## Attribution

This implementation follows the ray-tracing methodology described in:

> Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017). Fault slip distribution of the 2016 Fukushima earthquake estimated from tsunami waveforms. *Pure and Applied Geophysics*, 174(8), 2925–2943. https://doi.org/10.1007/s00024-017-1590-2

## Files

| File | Description |
|------|-------------|
| `rt_rungekutta_sp.py` | Core RK4 integrator for a single ray. Implements the spherical equations of motion and terminates at boundaries, shallow water, or land. |
| `raytracing.py` | Fans rays across all requested azimuths from a source point, builds the slowness field from bathymetry, and returns lon/lat/direction histories. |
| `main_raytracing_sp.py` | Driver script. Generates a synthetic peaks() bathymetry, traces a 360-ray azimuthal fan, and saves a plot. |
| `test_rk4.py` | Validation against analytical great-circle paths on a flat (uniform-depth) ocean. On a flat ocean, slowness gradients vanish and each ray must follow a great circle at constant speed. |

## Usage

```bash
# Run the synthetic example — produces raytracing_sp_c.png
python main_raytracing_sp.py

# Run the integrator validation
python test_rk4.py
```

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib
