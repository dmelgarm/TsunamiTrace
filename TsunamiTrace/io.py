"""
TsunamiTrace.io — bathymetry file loaders.

All loaders return arrays in the convention trace_rays expects:
  - depth  shape (n_lon, n_lat), positive values = ocean depth in metres
  - lon_arr / lat_arr  1-D, uniformly spaced, ascending

For matplotlib contour plots transpose depth: ``plt.contourf(lon, lat, depth.T)``.
"""
import numpy as np


def load_bathymetry(path, negate=True, comment='#'):
    """
    Load a three-column lon/lat/depth ASCII file onto a regular grid.

    The file may use any whitespace delimiter and may contain comment lines.
    The grid must be regular (uniformly spaced in both lon and lat); lon and
    lat need not have the same spacing.

    Parameters
    ----------
    path : str or path-like
        Path to the XYZ file.
    negate : bool, default True
        Most bathymetry files (GEBCO, ETOPO, SRTM30_PLUS, …) follow the
        geographic convention where ocean depths are **negative** and land
        elevations are positive.  When ``negate=True`` (the default) the
        depth column is negated before returning so that ocean depth is
        **positive** — the convention ``trace_rays`` expects.
        Pass ``negate=False`` if your file already stores ocean depth as a
        positive value.
    comment : str, default '#'
        Lines that start with this character are skipped.

    Returns
    -------
    lon_arr : ndarray, shape (n_lon,)
        Longitude values in degrees, ascending.
    lat_arr : ndarray, shape (n_lat,)
        Latitude values in degrees, ascending.
    depth : ndarray, shape (n_lon, n_lat)
        Depth in metres.  Positive = ocean, negative or zero = land/dry.
        First axis is longitude (lon_arr), second is latitude (lat_arr).

        For matplotlib ``contourf`` / ``contour`` transpose to row-major::

            plt.contourf(lon_arr, lat_arr, depth.T)

    Raises
    ------
    ValueError
        If the file does not form a regular grid (non-uniform spacing or
        missing grid points).

    Examples
    --------
    >>> lon, lat, depth = tt.load_bathymetry('data/cascadia.xyz')
    >>> depth_tracing = np.where(depth > 0, depth, 0.0)   # zero out land for trace_rays
    >>> ray_lon, ray_lat, _ = tt.trace_rays(lon, lat, depth_tracing, ...)
    """
    try:
        import pandas as pd
        data = pd.read_csv(
            path, comment=comment, sep=r'\s+', header=None,
            names=['lon', 'lat', 'depth'], engine='c',
        )
        lons   = data['lon'].to_numpy(dtype=float)
        lats   = data['lat'].to_numpy(dtype=float)
        depths = data['depth'].to_numpy(dtype=float)
    except ImportError:
        # Fallback: pure-numpy loader (slower but no pandas dependency)
        raw    = np.loadtxt(path, comments=comment)
        lons   = raw[:, 0]
        lats   = raw[:, 1]
        depths = raw[:, 2]

    if negate:
        depths = -depths

    # ── reconstruct the regular grid ─────────────────────────────────────────
    lon_arr = np.unique(lons)
    lat_arr = np.unique(lats)
    n_lon   = len(lon_arr)
    n_lat   = len(lat_arr)

    if len(lons) != n_lon * n_lat:
        raise ValueError(
            f"File does not form a regular grid: "
            f"found {len(lons):,} points but lon×lat = {n_lon}×{n_lat} = {n_lon*n_lat:,}.  "
            "Check for missing rows or duplicate coordinates."
        )

    # Verify uniform spacing in lon
    dlon = np.diff(lon_arr)
    if not np.allclose(dlon, dlon[0], rtol=1e-4):
        raise ValueError(
            "Longitude axis is not uniformly spaced "
            f"(min spacing {dlon.min():.6f}°, max {dlon.max():.6f}°)."
        )

    # Verify uniform spacing in lat
    dlat = np.diff(lat_arr)
    if not np.allclose(dlat, dlat[0], rtol=1e-4):
        raise ValueError(
            "Latitude axis is not uniformly spaced "
            f"(min spacing {dlat.min():.6f}°, max {dlat.max():.6f}°)."
        )

    # Reshape to (n_lon, n_lat) — sort by lon (slow) then lat (fast)
    order  = np.lexsort((lats, lons))
    depth  = depths[order].reshape(n_lon, n_lat)

    # Ensure lat axis is ascending inside each lon strip
    if lat_arr[0] > lat_arr[-1]:
        lat_arr = lat_arr[::-1]
        depth   = depth[:, ::-1]

    return lon_arr, lat_arr, depth
