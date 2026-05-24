"""
TsunamiTrace — tsunami ray tracing on a sphere.

Traces wavefront ray paths through variable-depth bathymetry using the
spherical Snell's-law equations integrated with 4th-order Runge-Kutta.

Basic usage
-----------
>>> import TsunamiTrace as tt
>>> ray_lon, ray_lat, ray_dir = tt.trace_rays(
...     lon_arr, lat_arr, depth,
...     dt=30, max_time=7200,
...     source_lon=157.0, source_lat=0.0,
...     azimuths_deg=np.arange(0, 360, 2),
... )

Reference
---------
Gusman, A. R., Satake, K., Shinohara, M., Sakai, S. I., & Tanioka, Y. (2017).
Fault slip distribution of the 2016 Fukushima earthquake estimated from
tsunami waveforms. Pure and Applied Geophysics, 174(8), 2925-2943.
https://doi.org/10.1007/s00024-017-1590-2
"""
from .raytracing import trace_rays
from . import io
from .io import load_bathymetry

__all__ = ["trace_rays", "load_bathymetry"]
__version__ = "0.1.0"
