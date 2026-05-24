"""
TsunamiTrace.io — bathymetry file loaders.

All loaders return arrays in the convention trace_rays expects:
  - depth    shape (n_lon, n_lat), positive values = ocean depth in metres
  - lon_arr / lat_arr  1-D, uniformly spaced, ascending

For matplotlib contour plots transpose depth: ``plt.contourf(lon, lat, depth.T)``.

Supported formats
-----------------
- Three-column ASCII (lon lat depth), any whitespace delimiter  (.xyz, .txt, …)
- NetCDF4 / NetCDF3  (.nc, .nc4)  — requires the ``netCDF4`` package
"""
import os
import numpy as np

# Variable names tried in order when auto-detecting the depth/elevation field
# in a NetCDF file.
_DEPTH_VAR_CANDIDATES = [
    'z', 'elevation', 'topo', 'topography', 'depth',
    'Band1', 'z_topo', 'altitude', 'bedrock',
]

# Coordinate name candidates
_LON_CANDIDATES = ['lon', 'longitude', 'x', 'X']
_LAT_CANDIDATES = ['lat', 'latitude', 'y', 'Y']


# ── public API ────────────────────────────────────────────────────────────────

def load_bathymetry(path, negate=True, comment='#', depth_var=None):
    """
    Load bathymetry from an ASCII XYZ file or a NetCDF file.

    The format is detected automatically from the file extension:
    ``.nc`` and ``.nc4`` are read as NetCDF; everything else is treated as a
    three-column whitespace-delimited ASCII file.

    Parameters
    ----------
    path : str or path-like
        Path to the bathymetry file.
    negate : bool, default True
        Most bathymetry products (GEBCO, ETOPO, SRTM30_PLUS, …) follow the
        geographic convention where ocean depths are **negative** and land
        elevations are positive.  When ``negate=True`` (the default) the
        depth values are negated before returning so that ocean depth is
        **positive**, which is the convention ``trace_rays`` expects.
        Pass ``negate=False`` if your file already stores ocean depth as a
        positive value.
    comment : str, default '#'
        (ASCII only) Lines beginning with this character are skipped.
    depth_var : str or None, default None
        (NetCDF only) Name of the variable that holds the depth/elevation
        data.  If ``None``, the loader tries a list of common names
        (``z``, ``elevation``, ``topo``, ``depth``, ``Band1``, …) and
        raises ``ValueError`` if none are found.

    Returns
    -------
    lon_arr : ndarray, shape (n_lon,)
        Longitude values in degrees, ascending.
    lat_arr : ndarray, shape (n_lat,)
        Latitude values in degrees, ascending.
    depth : ndarray, shape (n_lon, n_lat)
        Depth in metres.  Positive = ocean, negative or zero = land/dry.
        First axis is longitude, second is latitude.

        For matplotlib ``contourf`` / ``contour`` transpose to row-major::

            plt.contourf(lon_arr, lat_arr, depth.T)

    Raises
    ------
    ValueError
        If an ASCII file does not form a regular grid, or if a NetCDF file
        has no recognisable depth variable or coordinate variables.
    ImportError
        If a NetCDF file is requested but the ``netCDF4`` package is not
        installed.

    Examples
    --------
    >>> # ASCII — standard geographic convention (ocean negative)
    >>> lon, lat, depth = tt.load_bathymetry('data/cascadia.xyz')

    >>> # NetCDF — same convention, auto-detected depth variable
    >>> lon, lat, depth = tt.load_bathymetry('data/NE_pacific_4arcmin.nc')

    >>> # NetCDF — explicit depth variable name
    >>> lon, lat, depth = tt.load_bathymetry('data/custom.nc', depth_var='elevation')
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in ('.nc', '.nc4', '.netcdf', '.cdf'):
        return _load_netcdf(path, negate=negate, depth_var=depth_var)
    else:
        return _load_xyz(path, negate=negate, comment=comment)


# ── private loaders ───────────────────────────────────────────────────────────

def _load_xyz(path, negate, comment):
    """Load a three-column lon/lat/depth ASCII file."""
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

    # Reconstruct the regular grid
    lon_arr = np.unique(lons)
    lat_arr = np.unique(lats)
    n_lon   = len(lon_arr)
    n_lat   = len(lat_arr)

    if len(lons) != n_lon * n_lat:
        raise ValueError(
            f"File does not form a regular grid: "
            f"found {len(lons):,} points but lon x lat = "
            f"{n_lon} x {n_lat} = {n_lon * n_lat:,}.  "
            "Check for missing rows or duplicate coordinates."
        )

    dlon = np.diff(lon_arr)
    if not np.allclose(dlon, dlon[0], rtol=1e-4):
        raise ValueError(
            "Longitude axis is not uniformly spaced "
            f"(min {dlon.min():.6f}°, max {dlon.max():.6f}°)."
        )

    dlat = np.diff(lat_arr)
    if not np.allclose(dlat, dlat[0], rtol=1e-4):
        raise ValueError(
            "Latitude axis is not uniformly spaced "
            f"(min {dlat.min():.6f}°, max {dlat.max():.6f}°)."
        )

    # Reshape to (n_lon, n_lat): sort lon slow, lat fast
    order = np.lexsort((lats, lons))
    depth = depths[order].reshape(n_lon, n_lat)

    # Ensure lat is ascending
    if lat_arr[0] > lat_arr[-1]:
        lat_arr = lat_arr[::-1]
        depth   = depth[:, ::-1]

    return lon_arr, lat_arr, depth


def _load_netcdf(path, negate, depth_var):
    """Load bathymetry from a NetCDF3 or NetCDF4 file."""
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError(
            "Reading NetCDF files requires the netCDF4 package.  "
            "Install with:  conda install netCDF4  or  pip install netCDF4"
        )

    with nc.Dataset(path, 'r') as ds:
        # ── locate lon and lat coordinate variables ───────────────────────────
        lon_var = _find_coord(ds, _LON_CANDIDATES, axis_attr='X',
                              standard_names={'longitude'})
        lat_var = _find_coord(ds, _LAT_CANDIDATES, axis_attr='Y',
                              standard_names={'latitude'})
        if lon_var is None:
            raise ValueError(
                f"Cannot find a longitude coordinate in {path}.  "
                f"Variables present: {list(ds.variables.keys())}"
            )
        if lat_var is None:
            raise ValueError(
                f"Cannot find a latitude coordinate in {path}.  "
                f"Variables present: {list(ds.variables.keys())}"
            )

        lon_arr = np.array(ds.variables[lon_var][:], dtype=float)
        lat_arr = np.array(ds.variables[lat_var][:], dtype=float)

        # ── locate the depth/elevation variable ───────────────────────────────
        if depth_var is None:
            depth_var = _find_depth_var(ds, lon_var, lat_var)
        if depth_var is None:
            raise ValueError(
                f"Cannot find a depth/elevation variable in {path}.  "
                f"Tried: {_DEPTH_VAR_CANDIDATES}.  "
                f"Pass depth_var='<name>' to specify it explicitly.  "
                f"Variables present: {list(ds.variables.keys())}"
            )

        raw = ds.variables[depth_var][:]

    # Convert masked arrays (fill values) to plain numpy with NaN
    if hasattr(raw, 'filled'):
        raw = raw.filled(np.nan)
    depths = np.array(raw, dtype=float)

    if negate:
        depths = -depths

    # ── orient axes: ensure lon and lat are ascending ─────────────────────────
    if lon_arr[0] > lon_arr[-1]:
        lon_arr = lon_arr[::-1]
        # Figure out which axis of depths corresponds to lon
        # (handled below after axis detection)

    # ── map depths array axes to (n_lon, n_lat) ───────────────────────────────
    # NetCDF convention is typically (lat, lon) i.e. row-major geographic.
    # Use the shape to determine axis order.
    if depths.ndim != 2:
        raise ValueError(
            f"Depth variable '{depth_var}' has {depths.ndim} dimensions; "
            "expected 2 (lat, lon) or (lon, lat)."
        )

    n_lon, n_lat = len(lon_arr), len(lat_arr)

    if depths.shape == (n_lat, n_lon):
        # Standard (lat, lon) — transpose to (n_lon, n_lat)
        depth = depths.T
    elif depths.shape == (n_lon, n_lat):
        # Already (lon, lat)
        depth = depths
    else:
        raise ValueError(
            f"Depth variable '{depth_var}' has shape {depths.shape} which does "
            f"not match (n_lat={n_lat}, n_lon={n_lon}) or (n_lon, n_lat)."
        )

    # Ensure lat ascending; flip corresponding depth axis if needed
    if lat_arr[0] > lat_arr[-1]:
        lat_arr = lat_arr[::-1]
        depth   = depth[:, ::-1]

    if lon_arr[0] > lon_arr[-1]:
        lon_arr = lon_arr[::-1]
        depth   = depth[::-1, :]

    return lon_arr, lat_arr, depth


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_coord(ds, candidates, axis_attr, standard_names):
    """Return the name of the first matching coordinate variable."""
    # 1. Try candidate names directly
    for name in candidates:
        if name in ds.variables:
            return name
    # 2. Search by CF standard_name attribute
    for name, var in ds.variables.items():
        sn = getattr(var, 'standard_name', None)
        if sn and sn.lower() in standard_names:
            return name
    # 3. Search by axis attribute (CF convention: axis='X' or 'Y')
    for name, var in ds.variables.items():
        ax = getattr(var, 'axis', None)
        if ax and ax.upper() == axis_attr:
            return name
    return None


def _find_depth_var(ds, lon_var, lat_var):
    """Return the name of the depth/elevation variable by trying common names,
    then falling back to any 2-D variable whose dimensions match lon/lat."""
    # 1. Try common names
    for name in _DEPTH_VAR_CANDIDATES:
        if name in ds.variables:
            return name
    # 2. Any 2-D variable that is not a coordinate
    coord_names = {lon_var, lat_var}
    for name, var in ds.variables.items():
        if name not in coord_names and var.ndim == 2:
            return name
    return None
