"""
Defines custom callback functions used to compute various metrics at runtime.

"""
from .utility import *
from abc import ABC, abstractproperty, abstractmethod
import h5py
from collections import defaultdict
from .log import *
from firedrake import *
import numpy


class CallbackManager(defaultdict):
    """
    Stores callbacks in different categories and provides methods for
    evaluating them.

    Create callbacks and register them under ``'export'`` mode

    .. code-block:: python

        cb1 = VolumeConservation3DCallback(...)
        cb2 = TracerMassConservationCallback(...)
        cm = CallbackManager()
        cm.add(cb1, 'export')
        cm.add(cb2, 'export')

    Evaluate callbacks, calls :func:`evaluate` method of all callbacks
    registered in the given mode.

    .. code-block:: python

        cm.evaluate('export')

    """
    def __init__(self):
        super(CallbackManager, self).__init__(OrderedDict)

    def add(self, callback, mode):
        """
        Add a callback under the given mode

        :arg callback: a :class:`.DiagnosticCallback` object
        :arg str mode: register callback under this mode
        """
        key = callback.name
        self[mode][key] = callback

    def evaluate(self, mode, index=None):
        """
        Evaluate all callbacks registered under the given mode

        :arg str mode: evaluate all callbacks under this mode
        :kwarg int index: if provided, sets the export index. Default behavior
            is to append to the file or stream.
        """
        for key in sorted(self[mode]):
            self[mode][key].evaluate(index=index)


class DiagnosticHDF5(object):
    """
    A HDF5 file for storing diagnostic time series arrays.
    """
    @PETSc.Log.EventDecorator("thetis.DiagnosticHDF5.__init__")
    def __init__(self, filename, varnames, array_dim=1, attrs=None,
                 var_attrs=None, comm=COMM_WORLD, new_file=True,
                 dtype='d', include_time=True):
        """
        :arg str filename: Full filename of the HDF5 file.
        :arg varnames: List of variable names that the diagnostic callback
            provides
        :kwarg array_dim: Dimension of the output array.
            Can be a tuple for multi-dimensional output. Use "1" for scalars.
        :kwarg dict attrs: Global attributes to be saved in the hdf5 file.
        :kwarg dict var_attrs: nested dict of variable specific attributes,
             e.g. {'time': {'units': 'seconds since 1950-01-01'}}
        :kwarg comm: MPI communicator
        :kwarg bool new_file: Define whether to create a new hdf5 file or
            append to an existing one (if any)
        :kwarg dtype: array datatype
        :kwarg include_time: whether to include time array in the file
        """
        self.comm = comm
        self.filename = filename
        self.varnames = varnames
        self.nvars = len(varnames)
        self.array_dim = array_dim
        self.include_time = include_time
        if comm.rank == 0 and new_file:
            # create empty file with correct datasets
            with h5py.File(filename, 'w') as hdf5file:
                if include_time:
                    ds = hdf5file.create_dataset(
                        'time', (0, 1), maxshape=(None, 1), dtype=dtype)
                    if var_attrs is not None and 'time' in var_attrs:
                        ds.attrs.update(var_attrs['time'])
                dim_list = array_dim
                if isinstance(dim_list, tuple):
                    dim_list = list(dim_list)
                elif not isinstance(dim_list, list):
                    dim_list = list([dim_list])
                shape = tuple([0] + dim_list)
                max_shape = tuple([None] + dim_list)
                for var in self.varnames:
                    ds = hdf5file.create_dataset(
                        var, shape, maxshape=max_shape, dtype=dtype)
                    if var_attrs is not None and var in var_attrs:
                        ds.attrs.update(var_attrs[var])
                if attrs is not None:
                    hdf5file.attrs.update(attrs)

    def _expand_array(self, hdf5file, varname):
        """Expands array varname by 1 entry"""
        arr = hdf5file[varname]
        new_shape = list(arr.shape)
        new_shape[0] += 1
        arr.resize(tuple(new_shape))

    def _expand(self, hdf5file):
        """Expands data arrays by 1 entry"""
        for var in self.varnames:
            self._expand_array(hdf5file, var)
        if self.include_time:
            self._expand_array(hdf5file, 'time')

    def _nentries(self, hdf5file):
        return hdf5file[self.varnames[0]].shape[0]

    @PETSc.Log.EventDecorator("thetis.DiagnosticHDF5.export")
    def export(self, variables, time=None, index=None):
        """
        Appends a new entry of (time, variables) to the file.

        The HDF5 is updated immediately.

        :arg variables: values of entry
        :type variables: tuple of float
        :kwarg time: time stamp of entry
        :type time: float
        :kwarg int index: If provided, defines the time index in the file
        """
        if self.comm.rank == 0:
            with h5py.File(self.filename, 'a') as hdf5file:
                if index is not None:
                    nentries = self._nentries(hdf5file)
                    assert index <= nentries, 'time index out of range {:} <= {:} \n  in file {:}'.format(index, nentries, self.filename)
                    expand_required = index == nentries
                    ix = index
                if index is None or expand_required:
                    self._expand(hdf5file)
                    ix = self._nentries(hdf5file) - 1
                if self.include_time:
                    assert time is not None, 'time should be provided as 2nd argument to export()'
                    hdf5file['time'][ix] = time
                for i in range(self.nvars):
                    hdf5file[self.varnames[i]][ix, :] = variables[i]
                hdf5file.close()


class DiagnosticCallback(ABC):
    """
    A base class for all Callback classes
    """

    def __init__(self, solver_obj, array_dim=1, attrs=None,
                 outputdir=None,
                 export_to_hdf5=True,
                 append_to_log=True,
                 include_time=True,
                 hdf5_dtype='d',
                 start_time=None,
                 end_time=None):
        """
        :arg solver_obj: Thetis solver object
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg array_dim: Dimension of the output array.
            Can be a tuple for multi-dimensional output. Use "1" for scalars.
        :kwarg dict attrs: Global attributes to be saved in the hdf5 file.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, callback output messages will be
            printed in log
        :kwarg bool include_time: whether to include time in the hdf5 file
        :kwarg hdf5_dtype: Precision to use in hdf5 output: `d` for double
            precision (default), and `f` for single precision
        :kwarg start_time: Optional start time for callback evaluation
        :kwarg end_time: Optional end time for callback evaluation
        """
        if attrs is None:
            attrs = {}
        self.solver_obj = solver_obj
        self.outputdir = outputdir or self.solver_obj.options.output_directory
        self.array_dim = array_dim
        self.attrs = attrs
        self.var_attrs = {}
        self.append_to_hdf5 = export_to_hdf5
        self.append_to_log = append_to_log
        self.hdf5_dtype = hdf5_dtype
        self.include_time = include_time
        self._create_new_file = True
        self._hdf5_initialized = False
        self.start_time = start_time or -numpy.inf
        self.end_time = end_time or numpy.inf

        init_date = self.solver_obj.options.simulation_initial_date
        if init_date is not None and include_time:
            time_units = 'seconds since ' + init_date.isoformat()
            self.var_attrs['time'] = {'units': time_units}

    def set_write_mode(self, mode):
        """
        Define whether to create a new hdf5 file or append to an existing one

        :arg str mode: Either 'create' (default) or 'append'
        """
        assert mode in ['create', 'append']
        self._create_new_file = mode == 'create'

    def _create_hdf5_file(self):
        """
        Creates an empty hdf5 file with correct datasets.
        """
        if self.append_to_hdf5:
            comm = self.solver_obj.comm
            create_directory(self.outputdir, comm=comm)
            fname = 'diagnostic_{:}.hdf5'.format(self.name.replace(' ', '_'))
            fname = os.path.join(self.outputdir, fname)
            self.hdf_exporter = DiagnosticHDF5(fname, self.variable_names,
                                               array_dim=self.array_dim,
                                               new_file=self._create_new_file,
                                               attrs=self.attrs,
                                               var_attrs=self.var_attrs,
                                               comm=comm, dtype=self.hdf5_dtype,
                                               include_time=self.include_time)
        self._hdf5_initialized = True

    @abstractproperty
    def name(self):
        """The name of the diagnostic"""
        pass

    @abstractproperty
    def variable_names(self):
        """Names of all scalar values"""
        pass

    @abstractmethod
    def __call__(self):
        """
        Evaluate the diagnostic value.

        .. note::
            This method must implement all MPI reduction operations (if any).
        """
        pass

    @abstractmethod
    def message_str(self, *args):
        """
        A string representation.

        :arg args: If provided, these will be the return value from
            :meth:`__call__`.
        """
        return "{} diagnostic".format(self.name)

    def push_to_log(self, time, args):
        """
        Push callback status message to log

        :arg time: time stamp of entry
        :arg args: the return value from :meth:`__call__`.
        """
        print_output(self.message_str(*args))

    def push_to_hdf5(self, time, args, index=None):
        """
        Append values to HDF5 file.

        :arg time: time stamp of entry
        :arg args: the return value from :meth:`__call__`.
        """
        if not self._hdf5_initialized:
            self._create_hdf5_file()
        self.hdf_exporter.export(args, time=time, index=index)

    def evaluate(self, index=None):
        """
        Evaluates callback and pushes values to log and hdf file (if enabled)
        """
        time = self.solver_obj.simulation_time
        if time < self.start_time or time > self.end_time:
            return
        values = self.__call__()
        if self.append_to_log:
            self.push_to_log(time, values)
        if self.append_to_hdf5:
            self.push_to_hdf5(time, values, index=index)


class ScalarConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a scalar quantity"""
    variable_names = ['integral', 'relative_difference']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates scalar conservation check callback object

        :arg scalar_callback: Python function that takes the solver object as
            an argument and returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        super(ScalarConservationCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.initial_value = None

    def __call__(self):
        value = self.scalar_callback()
        if self.initial_value is None:
            self.initial_value = value
        rel_diff = (value - self.initial_value)/self.initial_value
        return value, rel_diff

    def message_str(self, *args):
        line = '{0:s} rel. error {1:11.4e}'.format(self.name, args[1])
        return line


class VolumeConservation3DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    name = 'volume3d'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        def vol3d():
            return comp_volume_3d(self.solver_obj.mesh)
        super(VolumeConservation3DCallback, self).__init__(vol3d, solver_obj, **kwargs)


class VolumeConservation2DCallback(ScalarConservationCallback):
    """Checks conservation of 2D volume (integral of water elevation field)"""
    name = 'volume2d'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """

        def vol2d():
            return comp_volume_2d(self.solver_obj.fields.elev_2d,
                                  self.solver_obj.fields.bathymetry_2d)
        super(VolumeConservation2DCallback, self).__init__(vol2d, solver_obj, **kwargs)


class TracerMassConservation2DCallback(ScalarConservationCallback):
    """
    Checks conservation of depth-averaged tracer mass

    Depth-averaged tracer mass is defined as the integral of 2D tracer
    multiplied by total depth.
    """
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            H = solver_obj.depth.get_total_depth(solver_obj.fields.elev_2d)
            return comp_tracer_mass_2d(solver_obj.fields[tracer_name], H)
        super(TracerMassConservation2DCallback, self).__init__(mass, solver_obj, **kwargs)


class ConservativeTracerMassConservation2DCallback(ScalarConservationCallback):
    """
    Checks conservation of conservative tracer mass which is depth integrated.
    """
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            # tracer is depth-integrated already, so just integrate over domain
            return assemble(solver_obj.fields[tracer_name]*dx)

        super(ConservativeTracerMassConservation2DCallback, self).__init__(mass, solver_obj, **kwargs)


class TracerMassConservationCallback(ScalarConservationCallback):
    """Checks conservation of total tracer mass"""
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            return comp_tracer_mass_3d(self.solver_obj.fields[tracer_name])
        super(TracerMassConservationCallback, self).__init__(mass, solver_obj, **kwargs)


class MinMaxConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a minimum/maximum"""
    variable_names = ['min_value', 'max_value', 'undershoot', 'overshoot']

    def __init__(self, minmax_callback, solver_obj, **kwargs):
        """
        :arg minmax_callback: Python function that takes the solver object as
            an argument and returns a (min, max) value tuple
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        super(MinMaxConservationCallback, self).__init__(solver_obj, **kwargs)
        self.minmax_callback = minmax_callback
        self.initial_value = None

    def __call__(self):
        value = self.minmax_callback()
        if self.initial_value is None:
            self.initial_value = value
        overshoot = max(value[1] - self.initial_value[1], 0.0)
        undershoot = min(value[0] - self.initial_value[0], 0.0)
        return value[0], value[1], undershoot, overshoot

    def message_str(self, *args):
        l = '{0:s} {1:g} {2:g}'.format(self.name, args[2], args[3])
        return l


class TracerOvershootCallBack(MinMaxConservationCallback):
    """Checks overshoots of the given tracer field."""
    name = 'tracer overshoot'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' overshoot'

        def minmax():
            tracer_min = self.solver_obj.fields[tracer_name].dat.data.min()
            tracer_max = self.solver_obj.fields[tracer_name].dat.data.max()
            tracer_min = self.solver_obj.comm.allreduce(tracer_min, op=MPI.MIN)
            tracer_max = self.solver_obj.comm.allreduce(tracer_max, op=MPI.MAX)
            return tracer_min, tracer_max
        super(TracerOvershootCallBack, self).__init__(minmax, solver_obj, **kwargs)


class DetectorsCallback(DiagnosticCallback):
    """
    Callback that evaluates the specified fields at the specified locations
    """
    def __init__(self, solver_obj,
                 detector_locations,
                 field_names,
                 name,
                 detector_names=None,
                 **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg detector_locations: List of x, y locations in which fields are to
            be interpolated.
        :arg field_names: List of fields to be interpolated.
        :arg name: Unique name for this callback and its associated set of
            detectors. This determines the name of the output h5 file
            (prefixed with `diagnostic_`).
        :arg detector_names: List of names for each of the detectors. If not
            provided automatic names of the form `detectorNNN` are created
            where NNN is the (0-padded) detector number.
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        # printing all detector output to log is probably not a useful default:
        kwargs.setdefault('append_to_log', False)
        self.field_dims = [solver_obj.fields[field_name].function_space().value_size
                           for field_name in field_names]
        attrs = {
            # use null-padded ascii strings, dtype='U' not supported in hdf5, see http://docs.h5py.org/en/latest/strings.html
            'field_names': numpy.array(field_names, dtype='S'),
            'field_dims': self.field_dims,
        }
        super().__init__(solver_obj, array_dim=sum(self.field_dims), attrs=attrs, **kwargs)

        ndetectors = len(detector_locations)
        if detector_names is None:
            fill = len(str(ndetectors))
            self.detector_names = ['detector{:0{fill}d}'.format(i, fill=fill) for i in range(ndetectors)]
        else:
            assert ndetectors == len(detector_names), "Different number of detector locations and names"
            self.detector_names = detector_names
        self._variable_names = self.detector_names
        self.detector_locations = detector_locations
        self.field_names = field_names
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def variable_names(self):
        return self.detector_names

    def _values_per_field(self, values):
        """
        Given all values evaulated in a detector location, return the values per field"""
        i = 0
        result = []
        for dim in self.field_dims:
            result.append(values[i:i+dim])
            i += dim
        return result

    def message_str(self, *args):
        return '\n'.join(
            'In {}: '.format(name) + ', '.join(
                '{}={}'.format(field_name, field_val) for field_name, field_val in zip(self.field_names, self._values_per_field(values)))
            for name, values in zip(self.detector_names, args))

    def _evaluate_field(self, field_name):
        return self.solver_obj.fields[field_name](self.detector_locations)

    def __call__(self):
        """
        Evaluate all current fields in all detector locations

        Returns a ndetectors x ndims array, where ndims is the sum of the
        dimension of the fields.
        """
        ndetectors = len(self.detector_locations)
        field_vals = []
        for field_name in self.field_names:
            field_vals.append(numpy.reshape(self._evaluate_field(field_name), (ndetectors, -1)))

        return numpy.hstack(field_vals)


class AccumulatorCallback(DiagnosticCallback):
    """
    Callback that evaluates a (scalar) functional involving integrals in both
    time and space.

    This callback can also be used to assemble time dependent objective
    functionals for adjoint simulations. Time integration is achieved using
    the trapezium rule.
    """
    variable_names = ['spatial integral at current timestep']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        :arg scalar_callback: Python function that returns a list of values of an objective functional.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)
        kwargs.setdefault('append_to_log', False)
        super(AccumulatorCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback      # Evaluate functional
        self.dt = solver_obj.options.timestep
        self.integrant = 0.
        self.old_value = None

    def __call__(self):
        scalar_value = self.scalar_callback()
        if self.old_value is not None:
            self.integrant += 0.5 * (self.old_value + scalar_value) * self.dt
        self.old_value = scalar_value

        return scalar_value

    def get_val(self):
        return self.integrant

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[0])
        return line


class TimeSeriesCallback2D(DiagnosticCallback):
    """
    Extract a time series of a 2D field at a given (x,y) location

    Currently implemented only for the 3D model.
    """
    name = 'timeseries'
    variable_names = ['value']

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback2D.__init__")
    def __init__(self, solver_obj, fieldnames, x, y,
                 location_name, z=None,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True,
                 tolerance=1e-3, eval_func=None,
                 start_time=None, end_time=None):
        """
        :arg solver_obj: Thetis :class:`FlowSolver` object
        :arg fieldnames: List of fields to extract
        :arg x, y: location coordinates in model coordinate system.
        :arg location_name: Unique name for this location. This
            determines the name of the output h5 file (prefixed with
            `diagnostic_`).
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, callback output messages will be
            printed in log
        :kwarg start_time: Optional start time for timeseries extraction
        :kwarg end_time: Optional end time for timeseries extraction
        """
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = self.location_name
        self.on_sphere = solver_obj.mesh2d.geometric_dimension() == 3
        if self.on_sphere:
            assert z is not None, 'z coordinate must be defined on a manifold mesh'
            attrs['z'] = z
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
        super(TimeSeriesCallback2D, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=1,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log,
            start_time=start_time,
            end_time=end_time)
        self.x = x
        self.y = y
        self.z = z
        self.tolerance = tolerance
        self.eval_func = eval_func
        self._initialized = False

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback2D._initialize")
    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # construct mesh points
        xyz = (self.x, self.y, self.z) if self.on_sphere else (self.x, self.y)
        self.xyz = numpy.array([xyz])

        # test evaluation
        try:
            if self.eval_func is None:
                self.solver_obj.fields.bathymetry_2d.at(self.xyz, tolerance=self.tolerance)
            else:
                self.eval_func(self.solver_obj.fields.bathymetry_2d, self.xyz, tolerance=self.tolerance)
        except PointNotInDomainError as e:
            error(
                '{:}: Station "{:}" out of horizontal domain'.format(
                    self.__class__.__name__, self.location_name)
            )
            raise e
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback2D.__call__")
    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                if self.eval_func is None:
                    val = field.at(self.xyz, tolerance=self.tolerance)
                else:
                    val = self.eval_func(field, self.xyz, tolerance=self.tolerance)
                arr = numpy.array(val)
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, value in zip(self.fieldnames, args):
            out += 'Evaluated {:} at {:}: {:.3g}\n'.format(
                fieldname, self.location_name, value[0])
        out = out[:-1]  # suppress last line break
        return out


class TimeSeriesCallback3D(DiagnosticCallback):
    """
    Extract a time series of a 3D field at a given (x,y,z) location

    Currently implemented only for the 3D model.
    """
    name = 'timeseries'
    variable_names = ['value']

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback3D.__init__")
    def __init__(self, solver_obj, fieldnames, x, y, z,
                 location_name,
                 outputdir=None, export_to_hdf5=True, append_to_log=True,
                 start_time=None, end_time=None):
        """
        :arg solver_obj: Thetis :class:`FlowSolver` object
        :arg fieldnames: List of fields to extract
        :arg x, y, z: location coordinates in model coordinate system.
        :arg location_name: Unique name for this location. This
            determines the name of the output h5 file (prefixed with
            `diagnostic_`).
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, callback output messages will be
            printed in log
        :kwarg start_time: Optional start time for timeseries extraction
        :kwarg end_time: Optional end time for timeseries extraction
        """
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y, 'z': z}
        attrs['location_name'] = self.location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
        super(TimeSeriesCallback3D, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=1,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log,
            start_time=start_time,
            end_time=end_time)
        self.x = x
        self.y = y
        self.z = z
        self._initialized = False

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback3D._initialize")
    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        try:
            min_z = -self.solver_obj.fields.bathymetry_2d.at((self.x, self.y))
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e

        if self.z < min_z:
            new_z = min_z + 0.1
            warning('Water depth too shallow at {:}; replacing z={:} by z={:}'.format(self.location_name, self.z, new_z))
            self.z = new_z

        # construct mesh points
        self.xyz = numpy.array([[self.x, self.y, self.z]])
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.TimeSeriesCallback3D.__call__")
    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = numpy.array(field.at(self.xyz))
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, value in zip(self.fieldnames, args):
            out += 'Evaluated {:} at {:} {:.2f} m: {:.3g}\n'.format(
                fieldname, self.location_name, self.z, value[0])
        out = out[:-1]  # suppress last line break
        return out


class VerticalProfileCallback(DiagnosticCallback):
    """
    Extract a vertical profile of a 3D field at a given (x,y) location

    Only for the 3D model.
    """
    name = 'vertprofile'
    variable_names = ['z_coord', 'value']

    @PETSc.Log.EventDecorator("thetis.VerticalProfileCallback.__init__")
    def __init__(self, solver_obj, fieldnames, x, y,
                 location_name,
                 npoints=48,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        """
        :arg solver_obj: Thetis :class:`FlowSolver` object
        :arg fieldnames: List of fields to extract
        :arg x, y: location coordinates in model coordinate system.
        :arg location_name: Unique name for this location. This
            determines the name of the output h5 file (prefixed with
            `diagnostic_`).
        :arg int npoints: Number of points along the vertical axis. The 3D
            field will be interpolated on these points, ranging from the bottom
            to the (time dependent) free surface.
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, callback output messages will be
            printed in log
        """
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = self.location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = ['z_coord'] + field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
        super(VerticalProfileCallback, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=npoints,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x = x
        self.y = y
        self.npoints = npoints
        self.xy = numpy.array([self.x, self.y])
        self.xyz = numpy.zeros((self.npoints, 3))
        self.xyz[:, 0] = self.x
        self.xyz[:, 1] = self.y
        self.epsilon = 1e-2  # nudge points to avoid libspatialindex errors
        self.alpha = numpy.linspace(0, 1, self.npoints)
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.VerticalProfileCallback._construct_z_array")
    def _construct_z_array(self):
        # construct mesh points for func evaluation
        try:
            depth = self.solver_obj.fields.bathymetry_2d.at(self.xy)
            elev = self.solver_obj.fields.elev_cg_2d.at(self.xy)
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e
        z_min = -(depth - self.epsilon)
        z_max = elev - self.epsilon
        self.xyz[:, 2] = z_max + (z_min - z_max)*self.alpha

    @PETSc.Log.EventDecorator("thetis.VerticalProfileCallback.__call__")
    def __call__(self):
        if not self._initialized:
            self._initialize()
        # update time-dependent z array
        self._construct_z_array()

        outvals = [self.xyz[:, 2]]
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = numpy.array(field.at(self.xyz))
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, prof in zip(self.fieldnames, args[1:]):
            minval = prof.min()
            maxval = prof.max()
            out += 'Evaluated {:} profile at {:}: range {:.3g} - {:.3g}\n'.format(
                fieldname, self.location_name, minval, maxval)
        out = out[:-1]  # suppress last line break
        return out


class TransectCallback(DiagnosticCallback):
    """
    Extract a vertical transect of a 3D field at a given (x,y) locations.

    Only for the 3D model.
    """
    name = 'transect'
    variable_names = ['z_coord', 'value']

    @PETSc.Log.EventDecorator("thetis.TransectCallback.__init__")
    def __init__(self, solver_obj, fieldnames, x, y,
                 location_name,
                 n_points_z=48,
                 z_min=None, z_max=None,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        """
        :arg solver_obj: Thetis :class:`FlowSolver` object
        :arg fieldnames: List of fields to extract
        :arg x, y: location coordinates in model coordinate system.
        :arg location_name: Unique name for this location. This
            determines the name of the output h5 file (prefixed with
            `diagnostic_`).
        :arg int n_points_z: Number of points along the vertical axis. The 3D
            field will be interpolated on these points, ranging from the bottom
            to the (time dependent) free surface.
        :kwarg float z_min, zmax: Force min/max value of z coordinate extent.
            By default, transect will cover entire depth from bed to surface.
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, will print extracted min/max values
            of each field to log
        """
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.name += '_' + self.location_name
        self.name += '_' + field_str

        self.x = numpy.array([x]).ravel()
        self.y = numpy.array([y]).ravel()
        if len(self.x) == 1:
            self.x = numpy.ones_like(self.y) * self.x
        if len(self.y) == 1:
            self.y = numpy.ones_like(self.x) * self.y

        attrs = {'x': self.x, 'y': self.y}
        attrs['location_name'] = self.location_name

        assert len(self.y) == len(self.x)
        self.n_points_xy = len(self.x)
        self.n_points_z = n_points_z
        self.value_shape = (self.n_points_z, self.n_points_xy)
        self.force_z_min = z_min
        self.force_z_max = z_max

        self.field_dims = {}
        for f in self.fieldnames:
            func = solver_obj.fields[f]
            self.field_dims[f] = func.function_space().value_size
        self.variable_names = ['z_coord']
        for f, f_short in zip(fieldnames, field_short_names):
            if self.field_dims[f] == 1:
                self.variable_names.append(f_short)
            else:
                coords = ['x', 'y', 'z']
                for k in range(self.field_dims[f]):
                    f_comp_name = f_short + '_' + coords[k]
                    self.variable_names.append(f_comp_name)

        super().__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=self.value_shape,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self._initialized = False

    @PETSc.Log.EventDecorator("thetis.TransectCallback._initialize")
    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # construct mesh points for evaluation
        self.xy = list(zip(self.x, self.y))
        self.trans_x = numpy.tile(self.x[numpy.newaxis, :], (self.n_points_z, 1))
        self.trans_y = numpy.tile(self.y[numpy.newaxis, :], (self.n_points_z, 1))

    @PETSc.Log.EventDecorator("thetis.TransectCallback._update_coords")
    def _update_coords(self):
        try:
            depth = numpy.array(self.solver_obj.fields.bathymetry_2d.at(self.xy))
            elev = numpy.array(self.solver_obj.fields.elev_cg_2d.at(self.xy))
        except PointNotInDomainError as e:
            error('{:}: Transect "{:}" point out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e
        epsilon = 1e-5  # nudge points to avoid libspatialindex errors
        z_min = -(depth - epsilon)
        z_max = elev - epsilon
        if self.force_z_min is not None:
            z_min = numpy.maximum(z_min, self.force_z_min)
        if self.force_z_max is not None:
            z_max = numpy.minimum(z_max, self.force_z_max)
        self.trans_z = numpy.linspace(z_max, z_min, self.n_points_z)
        self.trans_z = self.trans_z.reshape(self.value_shape)
        self.xyz = numpy.vstack((self.trans_x.ravel(),
                                 self.trans_y.ravel(),
                                 self.trans_z.ravel())).T

    @PETSc.Log.EventDecorator("thetis.TransectCallback.__call__")
    def __call__(self):
        if not self._initialized:
            self._initialize()
        self._update_coords()

        outvals = [self.trans_z]
        for fieldname in self.fieldnames:
            field = self.solver_obj.fields[fieldname]
            field_dim = self.field_dims[fieldname]
            try:
                vals = field.at(tuple(self.xyz))
                arr = numpy.array(vals)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data on transect {:}'.format(self.__class__.__name__, self.location_name))
                raise e
            # arr has shape (nxy, nz, ncomponents)
            shape = list(self.value_shape) + [field_dim]
            arr = numpy.array(vals).reshape(shape)
            # convert to list of components [(nxy, nz) , ...]
            components = [arr[..., i] for i in range(arr.shape[-1])]
            outvals.extend(components)
        return tuple(outvals)

    def message_str(self, *args):
        out = 'Evaluated transect "{:}":\n'.format(self.location_name)
        lines = []
        for fieldname, arr in zip(self.variable_names[1:], args[1:]):
            minval = arr.min()
            maxval = arr.max()
            line = f'  {fieldname} range: {minval:.3g} - {maxval:.3g}'
            lines.append(line)
        out += '\n'.join(lines)
        return out
