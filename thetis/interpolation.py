"""
Methods for interpolating data from structured data sets on Thetis fields.


Simple example of an atmospheric pressure interpolator:

.. code-block:: python

    class WRFInterpolator(object):
        # Interpolates WRF atmospheric model data on 2D fields
        def __init__(self, function_space, atm_pressure_field, ncfile_pattern, init_date):
            self.atm_pressure_field = atm_pressure_field

            # object that interpolates forcing data from structured grid on the local mesh
            self.grid_interpolator = NetCDFLatLonInterpolator2d(function_space, coord_system)
            # reader object that can read fields from netCDF files, applies spatial interpolation
            self.reader = NetCDFSpatialInterpolator(self.grid_interpolator, ['prmsl'])
            # object that can find previous/next time stamps in a collection of netCDF files
            self.timesearch_obj = NetCDFTimeSearch(ncfile_pattern, init_date, NetCDFTimeParser)
            # finally a linear intepolator class that performs linar interpolation in time
            self.interpolator = LinearTimeInterpolator(self.timesearch_obj, self.reader)

        def set_fields(self, time):
            # Evaluates forcing fields at the given time
            pressure = self.interpolator(time)
            self.atm_pressure_field.dat.data_with_halos[:] = pressure


Usage:

.. code-block:: python

    atm_pressure_2d = Function(solver_obj.function_spaces.P1_2d, name='atm pressure')
    wrf_pattern = 'forcings/atm/wrf/wrf_air.2016_*_*.nc'
    wrf_atm = WRFInterpolator(
        solver_obj.function_spaces.P1_2d,
        wind_stress_2d, atm_pressure_2d, wrf_pattern, init_date)
    simulation_time = 3600.
    wrf_atm.set_fields(simulation_time)
"""
import glob
import os
from .timezone import *
from .log import *
from scipy.spatial import Delaunay, QhullError
import netCDF4
from abc import ABC, abstractmethod
from firedrake import *
from firedrake.petsc import PETSc
import re
import string
import numpy
import cftime

TIMESEARCH_TOL = 1e-6


def get_ncvar_name(ncfile, standard_name=None, long_name=None, var_name=None):
    """
    Look for variables that match either CF standard_name or long_name
    attributes.

    If both are defined, standard_name takes precedence.

    Note that the attributes in the netCDF file converted to
    lower case prior to checking.

    :arg ncfile: netCDF4 Dataset object
    :kwarg standard_name: a target standard_name, or a list of them
    :kwarg long_name: a target long_name, or a list of them
    :kwarg var_name: a target netCDF variable name, or a list of them
    """
    assert standard_name is not None or long_name is not None, \
        'Either standard_name or long_name must be defined'
    # convert to list

    def listify(arg):
        if not isinstance(arg, (list, tuple)):
            arg = [arg]
        if arg is None:
            arg = []
        return arg
    standard_name = listify(standard_name)
    long_name = listify(long_name)
    var_name = listify(var_name)
    found = False
    for name, var in ncfile.variables.items():
        if 'standard_name' in var.ncattrs():
            if var.standard_name.lower() in standard_name:
                found = True
                break
        if 'long_name' in var.ncattrs():
            if var.long_name.lower() in long_name:
                found = True
                break
        if var.name.lower() in var_name:
            found = True
            break

    if not found:
        filter_str = []
        if standard_name is not None:
            filter_str.append(f'standard_name={standard_name}')
        if long_name is not None:
            filter_str.append(f'long_name={long_name}')
        filter_str = ' '.join(filter_str)
        msg = f'Variable matching {filter_str} not found ' \
            f'in {ncfile.filepath()}'
        raise ValueError(msg)
    return name


class GridInterpolator(object):
    """
    A reuseable griddata interpolator object.

    Usage:

    .. code-block:: python

        interpolator = GridInterpolator(source_xyz, target_xyz)
        vals = interpolator(source_data)

    Example:

    .. code-block:: python

        x0 = numpy.linspace(0, 10, 10)
        y0 = numpy.linspace(5, 10, 10)
        X, Y = numpy.meshgrid(x, y)
        x = X.ravel(); y = Y.ravel()
        data = x + 25.*y
        x_target = numpy.linspace(1, 10, 20)
        y_target = numpy.linspace(5, 10, 20)
        interpolator = GridInterpolator(numpy.vstack((x, y)).T, numpy.vstack((target_x, target_y)).T)
        vals = interpolator(data)

    Based on
    http://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    """
    @PETSc.Log.EventDecorator("thetis.GridInterpolator.__init__")
    def __init__(self, grid_xyz, target_xyz, fill_mode=None, fill_value=numpy.nan,
                 normalize=False, dont_raise=False):
        """
        :arg grid_xyz: Array of source grid coordinates, shape (npoints, 2) or
            (npoints, 3)
        :arg target_xyz: Array of target grid coordinates, shape (n, 2) or
            (n, 3)
        :kwarg fill_mode: Determines how points outside the source grid will be
            treated. If 'nearest', value of the nearest source point will be
            used. Otherwise a constant fill value will be used (default).
        :kwarg float fill_value: Set the fill value (default: NaN)
        :kwarg bool normalize: If true the data is scaled to unit cube before
            interpolation. Default: False.
        :kwarg bool dont_raise: Do not raise a Qhull error if triangulation
            fails. In this case the data will be set to fill value or nearest
            neighbor value.
        """
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.normalize = normalize
        self.fill_nearest = self.fill_mode == 'nearest'
        self.shape = (target_xyz.shape[0], )
        ngrid_points = grid_xyz.shape[0]
        if self.fill_nearest:
            assert ngrid_points > 0, 'at least one source point is needed'
        if self.normalize:

            def get_norm_params(x, scale=None):
                min = x.min()
                max = x.max()
                if scale is None:
                    scale = max - min
                a = 1./scale
                b = -min*a
                return a, b

            ax, bx = get_norm_params(target_xyz[:, 0])
            ay, by = get_norm_params(target_xyz[:, 1])
            az, bz = get_norm_params(target_xyz[:, 2])
            self.norm_a = numpy.array([ax, ay, az])
            self.norm_b = numpy.array([bx, by, bz])

            ngrid_xyz = self.norm_a*grid_xyz + self.norm_b
            ntarget_xyz = self.norm_a*target_xyz + self.norm_b
        else:
            ngrid_xyz = grid_xyz
            ntarget_xyz = target_xyz

        self.cannot_interpolate = False
        try:
            d = ngrid_xyz.shape[1]
            tri = Delaunay(ngrid_xyz)
            # NOTE this becomes expensive in 3D for npoints > 10k
            simplex = tri.find_simplex(ntarget_xyz)
            vertices = numpy.take(tri.simplices, simplex, axis=0)
            temp = numpy.take(tri.transform, simplex, axis=0)
            delta = ntarget_xyz - temp[:, d]
            bary = numpy.einsum('njk,nk->nj', temp[:, :d, :], delta)
            self.vtx = vertices
            self.wts = numpy.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
            self.outside = numpy.any(~numpy.isfinite(self.wts), axis=1)
            self.outside += numpy.any(self.wts < 0, axis=1)
            self.outside = numpy.nonzero(self.outside)[0]
            self.fill_nearest *= len(self.outside) > 0
            if self.fill_nearest:
                # find nearest neighbor in the data set
                from scipy.spatial import cKDTree
                dist, ix = cKDTree(ngrid_xyz).query(ntarget_xyz[self.outside])
                self.outside_to_nearest = ix
        except QhullError as e:
            if not dont_raise:
                raise e
            self.cannot_interpolate = True
            if self.fill_nearest:
                # find nearest neighbor in the data set
                from scipy.spatial import cKDTree
                dist, ix = cKDTree(ngrid_xyz).query(ntarget_xyz)
                self.outside_to_nearest = ix

    @PETSc.Log.EventDecorator("thetis.GridInterpolator.__call__")
    def __call__(self, values):
        """
        Interpolate values defined on grid_xyz to target_xyz.

        :arg values: Array of source values to interpolate, shape (npoints, )
        :kwarg float fill_value: Fill value to use outside the source grid (default: NaN)
        """
        if self.cannot_interpolate:
            if self.fill_nearest:
                ret = values[self.outside_to_nearest]
            else:
                ret = numpy.ones(self.shape)*self.fill_value
        else:
            ret = numpy.einsum('nj,nj->n', numpy.take(values, self.vtx), self.wts)
            if self.fill_nearest:
                ret[self.outside] = values[self.outside_to_nearest]
            else:
                ret[self.outside] = self.fill_value
        return ret


class FileTreeReader(object):
    """
    Abstract base class of file tree reader object
    """
    @abstractmethod
    def __call__(self, filename, time_index):
        """
        Reads a data for one time step from the file

        :arg str filename: a filename where to find the data (e.g. filename)
        :arg int time_index: time index to read
        :return: a list of floats or numpy.array_like objects
        """
        pass


class NetCDFTimeSeriesReader(FileTreeReader):
    """
    A simple netCDF reader that returns a time slice of the given variable.

    This class does not interpolate the data in any way. Useful for
    interpolating time series.
    """
    def __init__(self, variable_list, time_variable_name='time'):
        self.variable_list = variable_list
        self.time_variable_name = time_variable_name
        self.time_dim = None
        self.ndims = None

    def _detect_time_dim(self, ncfile):
        assert self.time_variable_name in ncfile.dimensions
        nc_var = ncfile[self.variable_list[0]]
        assert self.time_variable_name in nc_var.dimensions
        self.time_dim = nc_var.dimensions.index(self.time_variable_name)
        self.ndims = len(nc_var.dimensions)

    def _get_slice(self, time_index):
        """
        Returns a slice object that extracts a single time index
        """
        if self.ndims == 1:
            return time_index
        slice_list = [slice(None, None, None)]*self.ndims
        slice_list[self.time_dim] = slice(time_index, time_index+1, None)
        return slice_list

    def __call__(self, filename, time_index):
        """
        Reads a time_index from the data base

        :arg str filename: netcdf file where to find the data
        :arg int time_index: time index to read
        :return: a float or numpy.array_like value
        """
        assert os.path.isfile(filename), 'File not found: {:}'.format(filename)
        with netCDF4.Dataset(filename) as ncfile:
            if self.time_dim is None:
                self._detect_time_dim(ncfile)
            output = []
            for var in self.variable_list:
                values = ncfile[var][self._get_slice(time_index)]
                output.append(values)
            return output


def _get_subset_nodes(grid_x, grid_y, target_x, target_y):
    """
    Retuns grid nodes that are necessary for intepolating onto target_x,y
    """
    orig_shape = grid_x.shape
    grid_xy = numpy.array((grid_x.ravel(), grid_y.ravel())).T
    target_xy = numpy.array((target_x.ravel(), target_y.ravel())).T
    tri = Delaunay(grid_xy)
    simplex = tri.find_simplex(target_xy)
    vertices = numpy.take(tri.simplices, simplex, axis=0)
    nodes = numpy.unique(vertices.ravel())
    nodes_x, nodes_y = numpy.unravel_index(nodes, orig_shape)

    # x and y bounds for reading a subset of the netcdf data
    ind_x = slice(nodes_x.min(), nodes_x.max() + 1)
    ind_y = slice(nodes_y.min(), nodes_y.max() + 1)

    return nodes, ind_x, ind_y


class SpatialInterpolator(ABC):
    """
    Abstract base class for spatial interpolators that read data from disk
    """
    @abstractmethod
    def __init__(self, function_space, coord_system):
        """
        :arg function_space: target Firedrake FunctionSpace
        :arg coord_system: :class:`CoordinateSystem` object
        """
        pass

    @abstractmethod
    def interpolate(self, filename, variable_list, itime):
        """
        Interpolates data from the given file at given time step
        """
        pass


class SpatialInterpolator2d(SpatialInterpolator, ABC):
    """
    Abstract spatial interpolator class that can interpolate onto a 2D Function
    """
    @PETSc.Log.EventDecorator("thetis.SpatialInterpolator2d.__init__")
    def __init__(self, function_space, coord_system, fill_mode=None,
                 fill_value=numpy.nan):
        """
        :arg function_space: target Firedrake FunctionSpace
        :arg coord_system: :class:`CoordinateSystem` object
        :kwarg fill_mode: Determines how points outside the source grid will be
            treated. If 'nearest', value of the nearest source point will be
            used. Otherwise a constant fill value will be used (default).
        :kwarg float fill_value: Set the fill value (default: NaN)
        """
        assert function_space.ufl_element().value_shape == ()

        # construct local coordinates
        on_sphere = function_space.mesh().geometric_dimension() == 3

        if on_sphere:
            x, y, z = SpatialCoordinate(function_space.mesh())
            fsx = Function(function_space).interpolate(x).dat.data_with_halos
            fsy = Function(function_space).interpolate(y).dat.data_with_halos
            fsz = Function(function_space).interpolate(z).dat.data_with_halos
            coords = (fsx, fsy, fsz)
        else:
            x, y = SpatialCoordinate(function_space.mesh())
            fsx = Function(function_space).interpolate(x).dat.data_with_halos
            fsy = Function(function_space).interpolate(y).dat.data_with_halos
            coords = (fsx, fsy)

        lon, lat = coord_system.to_lonlat(*coords)
        self.mesh_lonlat = numpy.array([lon, lat]).T

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self._initialized = False

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolator2d._create_interpolator")
    def _create_interpolator(self, lat_array, lon_array):
        """
        Create compact interpolator by finding the minimal necessary support
        """
        assert len(lat_array.shape) == 2, 'Latitude must be two dimensional array.'
        assert len(lon_array.shape) == 2, 'longitude must be two dimensional array.'
        self.nodes, self.ind_lon, self.ind_lat = _get_subset_nodes(
            lon_array,
            lat_array,
            self.mesh_lonlat[:, 0],
            self.mesh_lonlat[:, 1]
        )

        subset_lat = lat_array[self.ind_lon, self.ind_lat].ravel()
        subset_lon = lon_array[self.ind_lon, self.ind_lat].ravel()
        subset_lonlat = numpy.array((subset_lon, subset_lat)).T
        self.grid_interpolator = GridInterpolator(
            subset_lonlat, self.mesh_lonlat, fill_mode=self.fill_mode,
            fill_value=self.fill_value)
        self._initialized = True

        # debug: plot subsets
        # import matplotlib.pyplot as plt
        # plt.plot(grid_lon_full, grid_lat_full, 'k.')
        # plt.plot(grid_lonlat[:, 0], grid_lonlat[:, 1], 'b.')
        # plt.plot(self.mesh_lonlat[:, 0], self.mesh_lonlat[:, 1], 'r.')
        # plt.show()

    @abstractmethod
    def interpolate(self, filename, variable_list, time):
        """
        Calls the interpolator object
        """
        pass


class NetCDFLatLonInterpolator2d(SpatialInterpolator2d):
    """
    Interpolates netCDF data on a local 2D unstructured mesh

    The intepolator is constructed for a single netCDF file that defines the
    source grid. Once the interpolator has been constructed, data can be read
    from any file that uses the same grid.

    This routine returns the data in numpy arrays.

    Usage:

    .. code-block:: python

        fs = FunctionSpace(...)
        myfunc = Function(fs, ...)
        ncinterp2d = NetCDFLatLonInterpolator2d(fs, coord_system, nc_filename)
        val1, val2 = ncinterp2d.interpolate(nc_filename, ['var1', 'var2'], 10)
        myfunc.dat.data_with_halos[:] = val1 + val2

    """
    @PETSc.Log.EventDecorator("thetis.NetCDFLatLonInterpolator2d.interpolate")
    def interpolate(self, nc_filename, variable_list, itime):
        """
        Interpolates data from a netCDF file onto Firedrake function space.

        :arg str nc_filename: netCDF file to read
        :arg variable_list: list of netCDF variable names to read
        :arg int itime: time index to read
        :returns: list of numpy.arrays corresponding to variable_list
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                name_lat = get_ncvar_name(
                    ncfile, 'latitude', 'latitude', ['latitude', 'lat'])
                name_lon = get_ncvar_name(
                    ncfile, 'longitude', 'longitude', ['longitude', 'lon'])
                grid_lat = ncfile[name_lat][:]
                grid_lon = ncfile[name_lon][:]
                lat_is_1d = len(grid_lat.shape) == 1
                lon_is_1d = len(grid_lat.shape) == 1
                assert lat_is_1d == lon_is_1d, 'Unsupported lat lon grid'
                if lat_is_1d and lon_is_1d:
                    grid_lon, grid_lat = numpy.meshgrid(grid_lon, grid_lat)
                self._create_interpolator(grid_lat, grid_lon)
            output = []
            for var in variable_list:
                msg = f'Variable {var} not found: {nc_filename}'
                assert var in ncfile.variables, msg
                # TODO generalize data dimensions, sniff from netcdf file
                grid_data = ncfile[var][itime, self.ind_lon, self.ind_lat].ravel()
                data = self.grid_interpolator(grid_data)
                output.append(data)
        return output


class NetCDFSpatialInterpolator(FileTreeReader):
    """
    Wrapper class that provides FileTreeReader API for grid interpolators
    """
    def __init__(self, grid_interpolator, variable_list):
        self.grid_interpolator = grid_interpolator
        self.variable_list = variable_list

    def __call__(self, filename, time_index):
        return self.grid_interpolator.interpolate(filename, self.variable_list, time_index)


class TimeParser(object):
    """
    Abstract base class for time definition objects.

    Defines the time span that a file (or data set) covers and provides a time
    index search routine.
    """
    @abstractmethod
    def get_start_time(self):
        """Returns the first time stamp in the file/data set"""
        pass

    @abstractmethod
    def get_end_time(self):
        """Returns the last time stamp in the file/data set"""
        pass

    @abstractmethod
    def find_time_stamp(self, t, previous=False):
        """
        Given time t, returns index of the next (previous) time stamp

        raises IndexError if t is out of range, i.e.
        t > self.get_end_time() or t < self.get_start_time()
        """
        pass


class NetCDFTimeParser(TimeParser):
    """
    Describes the time stamps stored in a netCDF file.
    """
    def __init__(self, filename, time_variable_name='time', allow_gaps=False,
                 verbose=False):
        """
        Construct a new object by scraping data from the given netcdf file.

        :arg str filename: name of the netCDF file to read
        :kwarg str time_variable_name: name of the time variable in the netCDF
            file (default: 'time')
        :kwarg bool allow_gaps: if False, an error is raised if time step is
            not constant.
        """
        self.filename = filename
        self.time_variable_name = time_variable_name

        def get_datetime(time, units, calendar):
            """
            Convert netcdf time value to datetime.
            """
            d = cftime.num2pydate(time, units, calendar)
            if d.tzinfo is None:
                d = pytz.utc.localize(d)  # assume UTC
            return d

        with netCDF4.Dataset(filename) as d:
            time_var = d[self.time_variable_name]
            assert 'units' in time_var.ncattrs(), \
                f'Time units not defined: {self.filename}'
            assert 'calendar' in time_var.ncattrs(), \
                f'Time calendar not defined: {self.filename}'
            units = time_var.units
            calendar = time_var.calendar

            dates = [get_datetime(t, units, calendar) for t in time_var[:]]
            self.time_array = numpy.array([datetime_to_epoch(d) for d in dates])
            self.start_time = epoch_to_datetime(float(self.time_array[0]))
            self.end_time = epoch_to_datetime(float(self.time_array[-1]))
            self.time_step = numpy.mean(numpy.diff(self.time_array))
            self.nb_steps = len(self.time_array)
            if verbose:
                print_output('Parsed file {:}'.format(filename))
                print_output('  Time span: {:} -> {:}'.format(self.start_time, self.end_time))
                print_output('  Number of time steps: {:}'.format(self.nb_steps))
                if self.nb_steps > 1:
                    print_output('  Time step: {:} h'.format(self.time_step/3600.))

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def find_time_stamp(self, t, previous=False):
        t_epoch = datetime_to_epoch(t) if isinstance(t, datetime.datetime) else t

        itime = numpy.searchsorted(self.time_array, t_epoch + TIMESEARCH_TOL)  # next
        if previous:
            itime -= 1
        if itime < 0:
            raise IndexError('Requested time out of bounds {:} < {:} in {:}'.format(t_epoch, self.time_array[0], self.filename))
        if itime >= len(self.time_array):
            raise IndexError('Requested time out of bounds {:} > {:} in {:}'.format(t_epoch, self.time_array[0], self.filename))
        return itime


class TimeSearch(object):
    """
    Base class for searching nearest time steps in a file tree or database
    """
    @abstractmethod
    def find(self, time, previous=False):
        """
        Find a next (previous) time stamp from a given time

        :arg float time: input time stamp
        :arg bool previous: if True, look for last time stamp before requested
            time. Otherwise returns next time stamp.
        :return: a (filename, time_index, time) tuple
        """
        pass


class NetCDFTimeSearch(TimeSearch):
    """
    Finds a nearest time stamp in a collection of netCDF files.
    """
    @PETSc.Log.EventDecorator("thetis.NetCDFTimeSearch.__init__")
    def __init__(self, file_pattern, init_date, netcdf_class, *args, **kwargs):
        all_files = glob.glob(file_pattern)
        assert len(all_files) > 0, 'No files found: {:}'.format(file_pattern)

        self.netcdf_class = netcdf_class
        self.init_date = init_date
        self.sim_start_time = datetime_to_epoch(self.init_date)
        self.verbose = kwargs.get('verbose', False)
        dates = []
        ncfiles = []
        for fn in all_files:
            nc = self.netcdf_class(fn, *args, **kwargs)
            ncfiles.append(nc)
            dates.append(nc.get_start_time())
        sort_ix = numpy.argsort(dates)
        self.files = numpy.array(all_files)[sort_ix]
        self.ncfiles = numpy.array(ncfiles)[sort_ix]
        self.start_datetime = numpy.array(dates)[sort_ix]
        self.start_times = [(s - self.init_date).total_seconds() for s in self.start_datetime]
        self.start_times = numpy.array(self.start_times)
        if self.verbose:
            print_output('{:}: Found time index:'.format(self.__class__.__name__))
            for i in range(len(self.files)):
                print_output('{:} {:} {:}'.format(i, self.files[i], self.start_times[i]))
                nc = self.ncfiles[i]
                print_output('  {:} -> {:}'.format(nc.start_time, nc.end_time))
                if nc.nb_steps > 1:
                    print_output('  {:} time steps, dt = {:} s'.format(nc.nb_steps, nc.time_step))
                else:
                    print_output('  {:} time steps'.format(nc.nb_steps))

    def simulation_time_to_datetime(self, t):
        return epoch_to_datetime(datetime_to_epoch(self.init_date) + t).astimezone(self.init_date.tzinfo)

    @PETSc.Log.EventDecorator("thetis.NetCDFTimeSearch.find")
    def find(self, simulation_time, previous=False):
        """
        Find file that contains the given simulation time

        :arg float simulation_time: simulation time in seconds
        :kwarg bool previous: if True finds previous existing time stamp instead
            of next (default False).
        :return: (filename, time index, simulation time) of found data
        """
        err_msg = 'No file found for time {:}'.format(self.simulation_time_to_datetime(simulation_time))
        ix = numpy.searchsorted(self.start_times, simulation_time + TIMESEARCH_TOL)
        if ix > 0:
            candidates = [ix-1, ix]
        else:
            candidates = [ix]
            if ix + 1 < len(self.start_times):
                candidates += [ix + 1]
        itime = None
        for i in candidates:
            try:
                nc = self.ncfiles[i]
                itime = nc.find_time_stamp(self.sim_start_time + simulation_time, previous=previous)
                time = nc.time_array[itime] - self.sim_start_time
                break
            except IndexError:
                pass
        if itime is None:
            raise Exception(err_msg)
        return self.files[i], itime, time


class DailyFileTimeSearch(TimeSearch):
    """
    Treats a list of daily files as a time series.

    File name pattern must be given as a string where the 4-digit year is
    tagged with "{year:04d}", and 2-digit zero-padded month and year are tagged
    with "{month:02d}" and "{day:02d}", respectively. The tags can be used
    multiple times.

    Example pattern:
        'ncom/{year:04d}/s3d.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc'

    In this time search method the time stamps are parsed solely from the
    filename, no other metadata is used. By default the data is assumed to be
    centered at 12:00 UTC every day.
    """
    @PETSc.Log.EventDecorator("thetis.DailyFileTimeSearch.__init__")
    def __init__(self, file_pattern, init_date, verbose=False,
                 center_hour=12, center_timezone=pytz.utc):
        self.file_pattern = file_pattern

        self.init_date = init_date
        self.sim_start_time = datetime_to_epoch(self.init_date)
        self.verbose = verbose

        all_files = self._find_files()
        dates = []
        for fn in all_files:
            d = self._parse_date(fn)
            timestamp = datetime.datetime(d['year'], d['month'], d['day'],
                                          center_hour, tzinfo=center_timezone)
            dates.append(timestamp)
        sort_ix = numpy.argsort(dates)
        self.files = numpy.array(all_files)[sort_ix]
        self.start_datetime = numpy.array(dates)[sort_ix]
        self.start_times = [(s - self.init_date).total_seconds() for s in self.start_datetime]
        self.start_times = numpy.array(self.start_times)
        if self.verbose:
            print_output('{:}: Found time index:'.format(self.__class__.__name__))
            for i in range(len(self.files)):
                print_output('{:} {:} {:}'.format(i, self.files[i], self.start_times[i]))
                print_output('  {:}'.format(self.start_datetime[i]))

    def _find_files(self):
        """Finds all files that match the given pattern."""
        search_pattern = str(self.file_pattern)
        search_pattern = search_pattern.replace(':02d}', ':}')
        search_pattern = search_pattern.replace(':04d}', ':}')
        search_pattern = search_pattern.format(year='*', month='*', day='*')
        all_files = glob.glob(search_pattern)
        assert len(all_files) > 0, 'No files found: {:}'.format(search_pattern)
        return all_files

    def _parse_date(self, filename):
        """
        Parse year, month, day from filename using the given pattern.
        """
        re_pattern = str(self.file_pattern)
        re_pattern = re_pattern.replace('{year:04d}', r'(\d{4,4})')
        re_pattern = re_pattern.replace('{month:02d}', r'(\d{2,2})')
        re_pattern = re_pattern.replace('{day:02d}', r'(\d{2,2})')
        o = re.findall(re_pattern, filename)
        assert len(o) == 1, 'parsing date from filename failed\n  {:}'.format(filename)
        values = [int(v) for v in o[0]]
        fmt = string.Formatter()
        labels = [s[1] for s in fmt.parse(self.file_pattern) if s[1] is not None]
        return dict(zip(labels, values))

    def simulation_time_to_datetime(self, t):
        return epoch_to_datetime(datetime_to_epoch(self.init_date) + t).astimezone(self.init_date.tzinfo)

    @PETSc.Log.EventDecorator("thetis.DailyFileTimeSearch.find")
    def find(self, simulation_time, previous=False):
        """
        Find file that contains the given simulation time

        :arg float simulation_time: simulation time in seconds
        :kwarg bool previous: if True finds previous existing time stamp instead
            of next (default False).
        :return: (filename, time index, simulation time) of found data
        """
        err_msg = 'No file found for time {:}'.format(self.simulation_time_to_datetime(simulation_time))
        ix = numpy.searchsorted(self.start_times, simulation_time + TIMESEARCH_TOL)
        i = ix - 1 if previous else ix
        assert i >= 0, err_msg
        assert i < len(self.start_times), err_msg
        itime = 0
        time = self.start_times[i]
        return self.files[i], itime, time


class LinearTimeInterpolator(object):
    """
    Interpolates time series in time

    User must provide timesearch_obj that finds time stamps from
    a file tree, and a reader that can read those time stamps into numpy arrays.

    Previous/next data sets are cached in memory to avoid hitting disk every
    time.
    """
    def __init__(self, timesearch_obj, reader):
        """
        :arg timesearch_obj: TimeSearch object
        :arg reader: FileTreeReader object
        """
        self.timesearch = timesearch_obj
        self.reader = reader
        self.cache = {}

    def _get_from_cache(self, key):
        """
        Fetch data set from cache, read if not present
        """
        if key not in self.cache:
            self.cache[key] = self.reader(key[0], key[1])
        return self.cache[key]

    def _clean_cache(self, keys_to_keep):
        """
        Remove cached data sets that are no longer needed
        """
        for key in list(self.cache.keys()):
            if key not in keys_to_keep:
                self.cache.pop(key)

    def __call__(self, t):
        """
        Interpolate at time t

        :retuns: list of numpy arrays
        """
        prev_id = self.timesearch.find(t, previous=True)
        next_id = self.timesearch.find(t, previous=False)

        prev = self._get_from_cache(prev_id)
        next = self._get_from_cache(next_id)
        self._clean_cache([prev_id, next_id])

        # interpolate
        t_prev = prev_id[2]
        t_next = next_id[2]
        alpha = (t - t_prev)/(t_next - t_prev)
        RELTOL = 1e-6
        assert alpha >= 0.0 - RELTOL and alpha <= 1.0 + RELTOL, \
            'Value {:} out of range {:} .. {:}'.format(t, t_prev, t_next)

        val = [(1.0 - alpha)*p + alpha*n for p, n in zip(prev, next)]
        return val


class NetCDFTimeSeriesInterpolator(object):
    """
    Reads and interpolates scalar time series from a sequence of netCDF files.
    """
    @PETSc.Log.EventDecorator("thetis.NetCDFTimeSeriesInterpolator.__init__")
    def __init__(self, ncfile_pattern, variable_list, init_date,
                 time_variable_name='time', scalars=None, allow_gaps=False):
        """
        :arg str ncfile_pattern: file search pattern, e.g. "mydir/foo_*.nc"
        :arg variable_list: list if netCDF variable names to read
        :arg datetime.datetime init_date: simulation start time
        :kwarg scalars: (optional) list of scalars; scale output variables by
            a factor.

        .. note::

            All the variables must have the same dimensions in the netCDF files.
            If the shapes differ, create separate interpolator instances.
        """
        self.reader = NetCDFTimeSeriesReader(
            variable_list, time_variable_name=time_variable_name)
        self.timesearch_obj = NetCDFTimeSearch(
            ncfile_pattern, init_date, NetCDFTimeParser,
            time_variable_name=time_variable_name, allow_gaps=allow_gaps)
        self.time_interpolator = LinearTimeInterpolator(self.timesearch_obj, self.reader)
        if scalars is not None:
            assert len(scalars) == len(variable_list)
        self.scalars = scalars

    @PETSc.Log.EventDecorator("thetis.NetCDFTimeSeriesInterpolator.__call__")
    def __call__(self, time):
        """
        Time series at the given time

        :returns: list of scalars or numpy.arrays
        """
        vals = self.time_interpolator(time)
        if self.scalars is not None:
            for i in range(len(vals)):
                vals[i] *= self.scalars[i]
        return vals
