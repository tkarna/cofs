"""
Generic methods for converting data between different spatial coordinate systems.
Uses pyproj library.
"""
import firedrake as fd
import pyproj
import numpy
from abc import ABC, abstractmethod
import warnings

LL_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84', errcheck=True)


class CoordinateSystem(ABC):
    """
    Base class for horizontal coordinate systems

    Provides methods for coordinate transformations etc.
    """
    @abstractmethod
    def to_lonlat(self, x, y):
        """Convert coordinates to latitude and longitude"""
        pass

    @abstractmethod
    def get_vector_rotator(self, x, y):
        """
        Returns a vector rotator object.

        The rotator converst vector-valued data to/from longitude, latitude
        coordinates.
        """
        pass


def proj_transform(x, y, trans):
    """
    Transform coordinates from source to target system.

    :arg x,y: coordinates, float or numpy.array_like
    :kwarg trans: pyproj Transformer object
    """
    x_is_array = isinstance(x, numpy.ndarray)
    y_is_array = isinstance(y, numpy.ndarray)
    numpy_inputs = x_is_array or y_is_array
    if numpy_inputs:
        assert x_is_array and y_is_array, 'both x and y must be numpy arrays'
        assert x.shape == y.shape, 'x and y must have same shape'
        # transform only non-nan entries as proj behavior can be erratic
        a = numpy.full_like(x, numpy.nan)
        b = numpy.full_like(y, numpy.nan)
        good_ix = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))
        a[good_ix], b[good_ix] = trans.transform(x[good_ix], y[good_ix])
    else:
        a, b = trans.transform(x, y)
    return a, b


class UTMCoordinateSystem(CoordinateSystem):
    """
    Represents Universal Transverse Mercator coordinate systems
    """
    def __init__(self, utm_zone, south=False):
        self.proj_obj = pyproj.Proj(proj='utm', zone=utm_zone, datum='WGS84',
                                    units='m', errcheck=True, south=south)
        self.transformer_lonlat = pyproj.Transformer.from_crs(
            self.proj_obj.srs, LL_WGS84.srs)
        self.transformer_xy = pyproj.Transformer.from_crs(
            LL_WGS84.srs, self.proj_obj.srs)

    def to_lonlat(self, x, y, positive_lon=False):
        """
        Convert (x, y) coordinates to (latitude, longitude)

        :arg x: x coordinate
        :arg y: y coordinate
        :type x: float or numpy.array_like
        :type y: float or numpy.array_like
        :kwarg positive_lon: should positive longitude be enforced?
        :return: longitude, latitude coordinates
        """
        lon, lat = proj_transform(x, y, trans=self.transformer_lonlat)
        if positive_lon:
            lon = numpy.mod(lon, 360.0)
        return lon, lat

    def to_xy(self, lon, lat):
        """
        Convert (latitude, longitude) coordinates to (x, y)

        :arg lon: longitude coordinate
        :arg lat: latitude coordinate
        :type longitude: float or numpy.array_like
        :type latitude: float or numpy.array_like
        :return: x, y coordinates
        """
        x, y = proj_transform(lon, lat, trans=self.transformer_xy)
        return x, y

    def get_mesh_lonlat_function(self, mesh2d):
        """
        Construct a :class:`Function` holding the mesh coordinates in
        longitude-latitude coordinates.

        :arg mesh2d: the 2D mesh
        """
        dim = mesh2d.topological_dimension()
        if dim != 2:
            raise ValueError(f'Expected a mesh of dimension 2, not {dim}')
        if mesh2d.geometric_dimension() != 2:
            raise ValueError('Mesh must reside in 2-dimensional space')
        x = mesh2d.coordinates.dat.data_ro[:, 0]
        y = mesh2d.coordinates.dat.data_ro[:, 1]
        lon, lat = self.transformer_lonlat.transform(x, y)
        lonlat = fd.Function(mesh2d.coordinates.function_space())
        lonlat.dat.data[:, 0] = lon
        lonlat.dat.data[:, 1] = lat
        return lonlat

    def get_vector_rotator(self, lon, lat):
        """
        Returns a vector rotator object.

        The rotator converts vector-valued data from longitude, latitude
        coordinates to mesh coordinate system.
        """
        return VectorCoordSysRotation(self.transformer_xy, lon, lat)


def get_vector_rotation_matrix(trans, x, y, delta=None):
    """
    Estimate rotation matrix that converts vectors defined in source_sys to
    target_sys. Conversion is carried out by the given `trans` object.

    Assume that we have a vector field defined in source_sys: vectors located at
    (x, y) define the x and y components. We can then rotate the vectors to
    represent x2 and y2 components of the target_sys by applying a local
    rotation:

    .. code-block:: python

        trans = pyproj.Transformer.from_crs(source_sys, target_sys)
        R, theta = get_vector_rotation_matrix(trans, x, y)
        v_xy = numpy.array([[v_x], [v_y]])
        v_new = numpy.matmul(R, v_xy)
        v_x2, v_y2 = v_new

    """
    if delta is None:
        delta = 1e-6  # ~1 m in LL_WGS84
    with warnings.catch_warnings():
        # supress spurious DeprecationWarning in numpy 1.25
        # (see https://github.com/pyproj4/pyproj/issues/1307)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        x1, y1 = trans.transform(x, y)
        x2, y2 = trans.transform(x, y + delta)
    dxdl = (x2 - x1) / delta
    dydl = (y2 - y1) / delta
    theta = numpy.arctan2(-dxdl, dydl)

    c = numpy.cos(theta)
    s = numpy.sin(theta)
    R = numpy.array([[c, -s], [s, c]])

    return R, theta


class VectorCoordSysRotation(object):
    """
    Rotates vectors defined in source_sys coordinates to a different coordinate
    system.

    """
    def __init__(self, trans, x, y):
        """
        :arg trans: pyproj.Transformer object, maps from source to destination system
        :arg x, y: coordinates in the source coordinate system
        """
        R, theta = get_vector_rotation_matrix(trans, x, y)
        self.rotation_sin = numpy.sin(theta)
        self.rotation_cos = numpy.cos(theta)

    def __call__(self, v_x, v_y, i_node=None):
        """
        Rotate vectors defined by the `v_x` and `v_y` components.

        :arg v_x, v_y: vector x, y components
        :kwarg ix_node: If not None, rotate the i-th vector instead of the
            whole array
        """
        # | c -s | | v_x |
        # | s  c | | v_y |
        f = [i_node] if i_node is not None else slice(None, None, None)
        u = v_x * self.rotation_cos[f] - v_y * self.rotation_sin[f]
        v = v_x * self.rotation_sin[f] + v_y * self.rotation_cos[f]
        return u, v
