"""
Routines for handling file exports.
"""
from .utility import *
from firedrake.output.vtk_output import is_cg, VTKFile
from collections import OrderedDict
import itertools


def is_2d(fs):
    """Tests wether a function space is 2D or 3D"""
    return fs.mesh().geometric_dimension() == 2


@PETSc.Log.EventDecorator("thetis.get_visu_space")
def get_visu_space(fs):
    """
    Returns an appropriate VTK visualization space for a function space

    :arg fs: function space
    :return: function space for VTK visualization
    """
    mesh = fs.mesh()
    family = 'Lagrange' if is_cg(fs) else 'Discontinuous Lagrange'
    if len(fs.ufl_element().value_shape) == 1:
        dim = fs.ufl_element().value_shape[0]
        visu_fs = get_functionspace(mesh, family, 1, family, 1,
                                    vector=True, dim=dim)
    elif len(fs.ufl_element().value_shape) == 2:
        visu_fs = get_functionspace(mesh, family, 1, family, 1,
                                    tensor=True)
    else:
        visu_fs = get_functionspace(mesh, family, 1, family, 1)
    # make sure that you always get the same temp work function
    visu_fs.max_work_functions = 1
    return visu_fs


class ExporterBase(object):
    """
    Base class for exporter objects.
    """
    def __init__(self, filename, outputdir, next_export_ix=0, verbose=False):
        """
        :arg string filename: output file name (without directory)
        :arg string outputdir: directory where file is stored
        :kwarg int next_export_ix: set the index for next output
        :kwarg bool verbose: print debug info to stdout
        """
        self.filename = filename
        self.outputdir = create_directory(outputdir)
        self.verbose = verbose
        # keeps track of export numbers
        self.next_export_ix = next_export_ix

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        self.next_export_ix = next_export_ix

    def export(self, function):
        """Exports given function to disk"""
        raise NotImplementedError('This method must be implemented in the derived class')


class VTKExporter(ExporterBase):
    """
    Class that handles Paraview VTK file exports
    """
    @PETSc.Log.EventDecorator("thetis.VTKExporter.__init__")
    def __init__(self, fs_visu, func_name, outputdir, filename,
                 next_export_ix=0, project_output=False, verbose=False):
        """
        :arg fs_visu: function space where input function will be cast
            before exporting
        :arg func_name: name of the function
        :arg outputdir: output directory
        :arg filename: name of the pvd file
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool project_output: project function to output space instead of
            interpolating
        :kwarg bool verbose: print debug info to stdout
        """
        super(VTKExporter, self).__init__(filename, outputdir, next_export_ix,
                                          verbose)
        self.fs_visu = fs_visu
        self.func_name = func_name
        self.project_output = project_output
        suffix = '.pvd'
        path = os.path.join(outputdir, filename)
        # append suffix if missing
        if (len(filename) < len(suffix)+1 or filename[:len(suffix)] != suffix):
            self.filename += suffix
        path = os.path.join(path, self.filename)
        self.outfile = VTKFile(path)
        self.cast_operators = {}

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        # NOTE vtk io objects store current export index not next
        super(VTKExporter, self).set_next_export_ix(next_export_ix)
        # FIXME hack to change correct output file count
        self.outfile.counter = itertools.count(start=self.next_export_ix)

    @PETSc.Log.EventDecorator("thetis.VTKExporter.export")
    def export(self, function):
        """Exports given function to disk"""
        assert self.fs_visu.max_work_functions == 1
        tmp_proj_func = self.fs_visu.get_work_function()
        # NOTE tmp function must be invariant as the projector is built only once
        op = self.cast_operators.get(function)
        if self.project_output:
            if op is None:
                op = Projector(function, tmp_proj_func)
                self.cast_operators[function] = op
            op.project()
        else:
            if op is None:
                op = Interpolator(function, tmp_proj_func)
                self.cast_operators[function] = op
            op.interpolate()
        # ensure correct output function name
        old_name = tmp_proj_func.name()
        tmp_proj_func.rename(name=self.func_name)
        self.outfile.write(tmp_proj_func, time=self.next_export_ix)
        self.next_export_ix += 1
        # restore old name
        tmp_proj_func.rename(name=old_name)
        self.fs_visu.restore_work_function(tmp_proj_func)


class HDF5Exporter(ExporterBase):
    """
    Stores fields in disk in native discretization using HDF5 containers
    """
    @PETSc.Log.EventDecorator("thetis.HDF5Exporter.__init__")
    def __init__(self, function_space, outputdir, filename_prefix,
                 next_export_ix=0, legacy_mode=False, verbose=False):
        """
        Create exporter object for given function.

        :arg function_space: space where the exported functions belong
        :type function_space: :class:`FunctionSpace`
        :arg string outputdir: directory where outputs will be stored
        :arg string filename_prefix: prefix of output filename. Filename is
            prefix_nnnnn.h5 where nnnnn is the export number.
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool legacy_mode: use legacy DumbCheckpoint format
        :kwarg bool verbose: print debug info to stdout
        """
        super(HDF5Exporter, self).__init__(filename_prefix, outputdir,
                                           next_export_ix, verbose)
        self.function_space = function_space
        self.dumb_checkpoint = legacy_mode

    def gen_filename(self, iexport):
        """
        Generate file name 'prefix_nnnnn.h5' for i-th export

        :arg int iexport: export index >= 0
        """
        filename = '{0:s}_{1:05d}'.format(self.filename, iexport)
        if not self.dumb_checkpoint:
            filename += '.h5'
        return os.path.join(self.outputdir, filename)

    def export_as_index(self, iexport, function):
        """
        Export function to disk using the specified export index number

        :arg int iexport: export index >= 0
        :arg function: :class:`Function` to export
        """
        assert function.function_space() == self.function_space, \
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print_output('saving {:} state to {:}'.format(function.name(), filename))
        if self.dumb_checkpoint:
            with DumbCheckpoint(filename, mode=FILE_CREATE, comm=function.comm) as f:
                f.store(function)
        else:
            with CheckpointFile(filename, 'w') as f:
                mesh = function.function_space().mesh()
                f.save_mesh(mesh)
                f.save_function(function)
        self.next_export_ix = iexport + 1

    @PETSc.Log.EventDecorator("thetis.HDF5Exporter.export")
    def export(self, function):
        """
        Export function to disk.

        Increments export index by 1.

        :arg function: :class:`Function` to export
        """
        self.export_as_index(self.next_export_ix, function)

    @PETSc.Log.EventDecorator("thetis.HDF5Exporter.load")
    def load(self, iexport, function):
        """
        Loads nodal values from disk and assigns to the given function

        :arg int iexport: export index >= 0
        :arg function: target :class:`Function`
        """
        assert function.function_space() == self.function_space, \
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print_output('loading {:} state from {:}'.format(function.name(), filename))
        if self.dumb_checkpoint:
            with DumbCheckpoint(filename, mode=FILE_READ, comm=function.comm) as f:
                f.load(function)
        else:
            with CheckpointFile(filename, 'r') as f:
                mesh = function.function_space().mesh()
                if not hasattr(mesh, 'sfXC'):
                    raise IOError('When loading fields from hdf5 checkpoint files, you should also read the mesh from checkpoint. See the documentation for `read_mesh_from_checkpoint()`')
                g = f.load_function(mesh, function.name())
                function.assign(g)


class ExportManager(object):
    """
    Helper object for exporting multiple fields simultaneously

    .. code-block:: python

        from .field_defs import field_metadata
        field_dict = {'elev_2d': Function(...), 'uv_3d': Function(...), ...}
        e = exporter.ExportManager('mydirectory',
                                   ['elev_2d', 'uv_3d', salt_3d'],
                                   field_dict,
                                   field_metadata,
                                   export_type='vtk')
        e.export()

    """
    def __init__(self, outputdir, fields_to_export, functions, field_metadata,
                 export_type='vtk', next_export_ix=0, verbose=False,
                 legacy_mode=False,
                 preproc_funcs={}):
        """
        :arg string outputdir: directory where files are stored
        :arg fields_to_export: list of fields to export
        :type fields_to_export: list of strings
        :arg functions: dict that contains all existing :class:`Function` s
        :arg field_metadata: dict of all field metadata.
            See :mod:`.field_defs`
        :kwarg str export_type: export format, either 'vtk' or 'hdf5'
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool verbose: print debug info to stdout
        :kwarg bool legacy_mode: use legacy `DumbCheckpoint` hdf5 format
        """
        self.outputdir = outputdir
        self.fields_to_export = fields_to_export
        # functions dict must be mutable for custom exports
        self.functions = {}
        self.functions.update(functions)
        self.field_metadata = field_metadata
        self.verbose = verbose
        self.preproc_callbacks = preproc_funcs
        # for each field create an exporter
        self.exporters = OrderedDict()
        for key in fields_to_export:
            field = self.functions.get(key)
            if field is not None and isinstance(field, Function):
                self.add_export(key, field, export_type,
                                legacy_mode=legacy_mode,
                                next_export_ix=next_export_ix)

    def add_export(self, fieldname, function,
                   export_type='vtk', next_export_ix=0, outputdir=None,
                   shortname=None, filename=None, legacy_mode=False,
                   preproc_func=None):
        """
        Adds a new field exporter in the manager.

        This method allows exporting both default Thetis fields and user
        defined fields. In the latter case the user must provide sufficient
        metadata, i.e. fieldname, shortname and filename.

        :arg string fieldname: canonical field name
        :arg function: Firedrake function to export
        :kwarg str export_type: export format, either 'vtk' or 'hdf5'
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg string outputdir: optional directory where files are stored
        :kwarg string shortname: override shortname defined in field_metadata
        :kwarg string filename: override filename defined in field_metadata
        :kwarg bool legacy_mode: use legacy `DumbCheckpoint` hdf5 format
        :kwarg preproc_func: optional funtion that will be called prior to
            exporting. E.g. for computing diagnostic fields.
        """
        if outputdir is None:
            outputdir = self.outputdir
        self.functions[fieldname] = function
        if shortname is None or filename is None:
            assert fieldname in self.field_metadata, \
                'Unknown field "{:}". For custom fields shortname and filename must be defined.'.format(fieldname)
        if shortname is None:
            shortname = self.field_metadata[fieldname]['shortname']
        if filename is None:
            filename = self.field_metadata[fieldname]['filename']
        field = self.functions.get(fieldname)
        if preproc_func is not None:
            self.preproc_callbacks[fieldname] = preproc_func
        if field is not None and isinstance(field, Function):
            native_space = field.function_space()
            visu_space = get_visu_space(native_space)
            if export_type.lower() == 'vtk':
                self.exporters[fieldname] = VTKExporter(visu_space, shortname,
                                                        outputdir, filename,
                                                        next_export_ix=next_export_ix)
            elif export_type.lower() == 'hdf5':
                self.exporters[fieldname] = HDF5Exporter(native_space,
                                                         outputdir, filename,
                                                         legacy_mode=legacy_mode,
                                                         next_export_ix=next_export_ix)

    def set_next_export_ix(self, next_export_ix):
        """Set export index to all child exporters"""
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(next_export_ix)

    def export(self):
        """
        Export all designated functions to disk

        Increments export index by 1.
        """
        if self.verbose and COMM_WORLD.rank == 0:
            sys.stdout.write('Exporting: ')
        for key in self.exporters:
            field = self.functions[key]
            if field is not None:
                if self.verbose and COMM_WORLD.rank == 0:
                    sys.stdout.write(key+' ')
                    sys.stdout.flush()
                if key in self.preproc_callbacks:
                    self.preproc_callbacks[key]()
                self.exporters[key].export(field)
        if self.verbose and COMM_WORLD.rank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

    @PETSc.Log.EventDecorator("thetis.ExportManager.export_bathymetry")
    def export_bathymetry(self, bathymetry_2d):
        """
        Special function to export 2D bathymetry data to disk

        :arg bathymetry_2d: 2D bathymetry :class:`Function`
        """
        bathfile = VTKFile(os.path.join(self.outputdir, 'init_bathymetry_2d/init_bathymetry_2d.pvd'))

        bathfile.write(bathymetry_2d)
