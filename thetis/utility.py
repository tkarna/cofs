"""
Utility functions and classes for 2D and 3D ocean models
"""
import os
import glob
import sys
from collections import OrderedDict, namedtuple  # NOQA

import ufl  # NOQA
from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI  # NOQA
from pyop2.profiling import timed_stage  # NOQA
import numpy
from functools import wraps
from pyadjoint.tape import no_annotations

from .field_defs import field_metadata
from .log import *
from .physical_constants import physical_constants

ds_surf = ds_t
ds_bottom = ds_b


class FrozenClass(object):
    """
    A class where creating a new attribute will raise an exception if
    :attr:`_isfrozen` is ``True``.

    :attr:`_unfreezedepth` allows for multiple applications of the
    ``unfrozen`` decorator.
    """
    _isfrozen = False
    _unfreezedepth = 0

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenClass, self).__setattr__(key, value)


def unfrozen(method):
    """
    Decorator to temporarily unfreeze an object
    whilst one of its methods is being called.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._isfrozen = False
        self._unfreezedepth += 1
        ret = method(self, *args, **kwargs)
        self._unfreezedepth -= 1
        self._isfrozen = self._unfreezedepth == 0
        return ret

    return wrapper


class SumFunction(object):
    """
    Helper class to keep track of sum of Coefficients.
    """
    def __init__(self):
        """
        Initialize empty sum.

        get operation returns Constant(0)
        """
        self.coeff_list = []

    def add(self, coeff):
        """
        Adds a coefficient to self
        """
        if coeff is None:
            return
        self.coeff_list.append(coeff)

    def get_sum(self):
        """
        Returns a sum of all added Coefficients
        """
        if len(self.coeff_list) == 0:
            return None
        return sum(self.coeff_list)


class AttrDict(dict):
    """
    Dictionary that provides both self['key'] and self.key access to members.

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """
    def __init__(self, *args, **kwargs):
        if sys.version_info < (2, 7, 4):
            raise Exception('AttrDict requires python >= 2.7.4 to avoid memory leaks')
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FieldDict(AttrDict):
    """
    AttrDict that checks that all added fields have proper meta data.

    Values can be either Function or Constant objects.
    """
    def _check_inputs(self, key, value):
        if key != '__dict__':
            from firedrake.functionspaceimpl import (MixedFunctionSpace,
                                                     WithGeometry)
            if not isinstance(value, (Function, Constant)):
                raise TypeError('Value must be a Function or Constant object')
            fs = value.function_space()
            is_mixed = (isinstance(fs, MixedFunctionSpace)
                        or (isinstance(fs, WithGeometry)
                            and isinstance(fs.topological, MixedFunctionSpace)))
            if not is_mixed and key not in field_metadata:
                msg = 'Trying to add a field "{:}" that has no metadata. ' \
                      'Add field_metadata entry to field_defs.py'.format(key)
                raise Exception(msg)

    def _set_functionname(self, key, value):
        """Set function.name to key to ensure consistent naming"""
        if isinstance(value, Function):
            value.rename(name=key)

    def __setitem__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setattr__(key, value)


def domain_constant(value, mesh, **kwargs):
    """Create constant over a domain

    Returns what used to be Constant(value, domain=mesh)"""
    try:
        shape = value.ufl_shape
        value = value.dat.data.flatten()
    except AttributeError:
        shape = np.shape(value)
        value = np.asarray(value).flatten()

    if len(shape) == 0:
        R = FunctionSpace(mesh, "R", 0)
    elif len(shape) == 1:
        R = VectorFunctionSpace(mesh, "R", 0, dim=shape[0])
    elif len(shape) == 2:
        R = TensorFunctionSpace(mesh, "R", 0, shape=shape)

    c = Function(R, **kwargs)
    c.assign(value)

    return c


def get_functionspace(mesh, h_family, h_degree, v_family=None, v_degree=None,
                      vector=False, tensor=False, hdiv=False, variant=None, v_variant=None,
                      **kwargs):
    cell_dim = mesh.cell_dimension()
    assert cell_dim in [2, (2, 1)], 'Unsupported cell dimension'
    hdiv_families = [
        'RT', 'RTF', 'RTCF', 'RAVIART-THOMAS',
        'BDM', 'BDMF', 'BDMCF', 'BREZZI-DOUGLAS-MARINI',
    ]
    if variant is None:
        if h_family.upper() in hdiv_families:
            if h_family in ['RTCF', 'BDMCF']:
                variant = 'equispaced'
            else:
                variant = 'integral'
        else:
            variant = 'equispaced'
    if v_variant is None:
        v_variant = 'equispaced'
    if cell_dim == (2, 1):
        if v_family is None:
            v_family = h_family
        if v_degree is None:
            v_degree = h_degree
        h_cell, v_cell = mesh.ufl_cell().sub_cells()
        h_elt = FiniteElement(h_family, h_cell, h_degree, variant=variant)
        v_elt = FiniteElement(v_family, v_cell, v_degree, variant=v_variant)
        elt = TensorProductElement(h_elt, v_elt)
        if hdiv:
            elt = HDiv(elt)
    else:
        elt = FiniteElement(h_family, mesh.ufl_cell(), h_degree, variant=variant)

    assert not (vector and tensor)
    constructor = TensorFunctionSpace if tensor else VectorFunctionSpace if vector else FunctionSpace
    return constructor(mesh, elt, **kwargs)


def get_extruded_base_element(ufl_element):
    """
    Return UFL TensorProductElement of an extruded UFL element.

    In case of a non-extruded mesh, returns the element itself.
    """
    if isinstance(ufl_element, firedrake.HDivElement):
        ufl_element = ufl_element._element
    if isinstance(ufl_element, firedrake.MixedElement):
        ufl_element = ufl_element.sub_elements[0]
    if isinstance(ufl_element, firedrake.VectorElement):
        ufl_element = ufl_element.sub_elements[0]  # take the first component
    if isinstance(ufl_element, firedrake.EnrichedElement):
        ufl_element = ufl_element._elements[0]
    return ufl_element


ElementContinuity = namedtuple("ElementContinuity", ["horizontal", "vertical"])
"""
A named tuple describing the continuity of an element in the horizontal/vertical direction.

The field value is one of "cg", "hdiv", or "dg".
"""


def element_continuity(ufl_element):
    """Return an :class:`ElementContinuity` instance with the
    continuity of a given element.

    :arg ufl_element: The UFL element to determine the continuity
        of.
    :returns: A new :class:`ElementContinuity` instance.
    """
    elem = ufl_element
    elem_types = {
        'Discontinuous Lagrange': 'dg',
        'Lagrange': 'cg',
        'Raviart-Thomas': 'hdiv',
        'RTCF': 'hdiv',
        'Brezzi-Douglas-Marini': 'hdiv',
        'BDMCF': 'hdiv',
        'Q': 'cg',
        'DQ': 'dg',
    }

    base_element = get_extruded_base_element(ufl_element)
    if isinstance(elem, firedrake.HDivElement):
        horiz_type = 'hdiv'
        vert_type = 'hdiv'
    elif isinstance(base_element, firedrake.TensorProductElement):
        a, b = base_element.sub_elements
        horiz_type = elem_types[a.family()]
        vert_type = elem_types[b.family()]
    else:
        horiz_type = elem_types[base_element.family()]
        vert_type = horiz_type
    return ElementContinuity(horiz_type, vert_type)


def create_directory(path, comm=COMM_WORLD):
    """
    Create a directory on disk

    Raises IOError if a file with the same name already exists.
    """
    if comm.rank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise IOError('file with same name exists', path)
        else:
            os.makedirs(path)
    comm.barrier()
    return path


def read_mesh_from_checkpoint(filename_or_outputdir, mesh_name=None):
    """
    Read mesh from a hdf5 checkpoint file.
    :arg filename_or_outputdir: file name of the hdf5 checkpoint file,
       or the output directory used in the previous run that create the
       checkpoint. In the latter case the mesh will be read from the
       first file that's found in outputdir/hdf5/*.h5

    When loading fields from a checkpoint file in a run, the mesh used in that
    run needs to be read from that same checkpoint file.

    When multiple checkpoint files are used as model inputs which have
    been created in separate runs/scripts, make sure that only one of those
    scripts created the original mesh (by reading a .msh file or using a mesh
    creation utility like RectangleMesh) and all other script read their mesh
    from the checkpoint created by that first script.  For example, a
    preprocessing script might read in a .msh file and interpolate and smoothen
    the bathymetry on that mesh and write out the result in a checkpoint. A
    first Thetis run should read its mesh from that checkpoint, and may then
    save a series of hdf5 files.  A second Thetis run can then read in the mesh
    and bathymetry from the checkpoint of the preprocessing script, and call
    load_state() to pick up from the previous run.
    """
    if os.path.isdir(filename_or_outputdir):
        # grab the first outputdir/hdf5/*.h5 file we can find
        path = os.path.join(filename_or_outputdir, 'hdf5', '*.h5')
        try:
            filename = next(glob.iglob(path))
        except StopIteration:
            raise IOError(f'No checkpoint files found in {path}')
    else:
        # assume we're being pointed to the hdf5 file directly
        filename = filename_or_outputdir

    with CheckpointFile(filename, 'r') as f:
        if mesh_name is None:
            mesh_name = 'firedrake_default'
        mesh = f.load_mesh(mesh_name)

    return mesh


@PETSc.Log.EventDecorator("thetis.get_facet_mask")
def get_facet_mask(function_space, facet='bottom'):
    """
    Returns the top/bottom nodes of extruded 3D elements.

    :arg function_space: Firedrake :class:`FunctionSpace` object
    :kwarg str facet: 'top' or 'bottom'

    .. note::
        The definition of top/bottom depends on the direction of the extrusion.
        Here we assume that the mesh has been extruded upwards (along positive
        z axis).
    """
    from tsfc.finatinterface import create_element as create_finat_element

    # get base element
    elem = get_extruded_base_element(function_space.ufl_element())
    assert isinstance(elem, TensorProductElement), \
        f'function space must be defined on an extruded 3D mesh: {elem}'
    # figure out number of nodes in sub elements
    h_elt, v_elt = elem.sub_elements
    nb_nodes_h = create_finat_element(h_elt).space_dimension()
    nb_nodes_v = create_finat_element(v_elt).space_dimension()
    # compute top/bottom facet indices
    # extruded dimension is the inner loop in index
    # on interval elements, the end points are the first two dofs
    offset = 0 if facet == 'bottom' else 1
    indices = numpy.arange(nb_nodes_h)*nb_nodes_v + offset
    return indices


@PETSc.Log.EventDecorator("thetis.extrude_mesh_sigma")
def extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, z_stretch_fact=1.0,
                       min_depth=None):
    """
    Extrudes a 2d surface mesh with bathymetry data defined in a 2d field.

    Generates a uniform terrain following mesh.

    :arg mesh2d: 2D mesh
    :arg n_layers: number of vertical layers
    :arg bathymetry: 2D :class:`Function` of the bathymetry
        (the depth of the domain; positive downwards)
    """
    mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)

    coordinates = mesh.coordinates
    fs_3d = coordinates.function_space()
    fs_2d = bathymetry_2d.function_space()
    new_coordinates = Function(fs_3d)

    z_stretch_func = Function(fs_2d)
    if isinstance(z_stretch_fact, Function):
        assert z_stretch_fact.function_space() == fs_2d
        z_stretch_func = z_stretch_fact
    else:
        z_stretch_func.assign(z_stretch_fact)

    # number of nodes in vertical direction
    n_vert_nodes = fs_3d.finat_element.space_dimension() / fs_2d.finat_element.space_dimension()

    min_depth_arr = numpy.ones((n_layers+1, ))*1e22
    if min_depth is not None:
        for i, v in enumerate(min_depth):
            min_depth_arr[i] = v

    nodes = get_facet_mask(fs_3d, 'bottom')
    idx = op2.Global(len(nodes), nodes, dtype=numpy.int32, name='node_idx')
    min_depth_op2 = op2.Global(len(min_depth_arr), min_depth_arr, name='min_depth')
    kernel = op2.Kernel("""
        void my_kernel(double *new_coords, double *old_coords, double *bath2d, double *z_stretch, int *idx, double *min_depth) {
            for ( int d = 0; d < %(nodes)d; d++ ) {
                double s_fact = z_stretch[d];
                for ( int e = 0; e < %(v_nodes)d; e++ ) {
                    new_coords[3*(idx[d]+e) + 0] = old_coords[3*(idx[d]+e) + 0];
                    new_coords[3*(idx[d]+e) + 1] = old_coords[3*(idx[d]+e) + 1];
                    double sigma = 1.0 - old_coords[3*(idx[d]+e) + 2]; // top 0, bot 1
                    double new_z = -bath2d[d] * pow(sigma, s_fact) ;
                    int layer = fmin(fmax(round(sigma*(%(n_layers)d + 1) - 1.0), 0.0), %(n_layers)d);
                    double max_z = -min_depth[layer];
                    new_z = fmax(new_z, max_z);
                    new_coords[3*(idx[d]+e) + 2] = new_z;
                }
            }
        }""" % {'nodes': fs_2d.finat_element.space_dimension(),
                'v_nodes': n_vert_nodes,
                'n_layers': n_layers},
        'my_kernel')

    op2.par_loop(kernel, mesh.cell_set,
                 new_coordinates.dat(op2.WRITE, fs_3d.cell_node_map()),
                 coordinates.dat(op2.READ, fs_3d.cell_node_map()),
                 bathymetry_2d.dat(op2.READ, fs_2d.cell_node_map()),
                 z_stretch_func.dat(op2.READ, fs_2d.cell_node_map()),
                 idx(op2.READ),
                 min_depth_op2(op2.READ),
                 iteration_region=op2.ALL)

    mesh.coordinates.assign(new_coordinates)

    return mesh


@PETSc.Log.EventDecorator("thetis.comp_volume_2d")
def comp_volume_2d(eta, bath):
    """Computes volume of the 2D domain as an integral of the elevation field"""
    val = assemble((eta+bath)*dx)
    return val


@PETSc.Log.EventDecorator("thetis.comp_volume_3d")
def comp_volume_3d(mesh):
    """Computes volume of the 3D domain as an integral"""
    val = assemble(Constant(1.0)*dx(domain=mesh))
    return val


@PETSc.Log.EventDecorator("thetis.comp_tracer_mass_2d")
def comp_tracer_mass_2d(scalar_func, total_depth):
    """
    Computes total tracer mass in the 2D domain
    :arg scalar_func: depth-averaged scalar :class:`Function` to integrate
    :arg total_depth: scalar UFL expression (e.g. from get_total_depth())
    """
    val = assemble(scalar_func*total_depth*dx)
    return val


@PETSc.Log.EventDecorator("thetis.comp_tracer_mass_3d")
def comp_tracer_mass_3d(scalar_func):
    """
    Computes total tracer mass in the 3D domain

    :arg scalar_func: scalar :class:`Function` to integrate
    """
    val = assemble(scalar_func*dx)
    return val


@PETSc.Log.EventDecorator("thetis.get_zcoord_from_mesh")
def get_zcoord_from_mesh(zcoord, solver_parameters=None):
    """
    Evaluates z coordinates from the 3D mesh

    :arg zcoord: scalar :class:`Function` where coordinates will be stored
    """
    # TODO coordinates should probably be interpolated instead
    if solver_parameters is None:
        solver_parameters = {}
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    l = fs.mesh().coordinates[2]*test*dx
    solve(a == l, zcoord, solver_parameters=solver_parameters)
    return zcoord


@PETSc.Log.EventDecorator("thetis.compute_baroclinic_head")
def compute_baroclinic_head(solver):
    r"""
    Computes the baroclinic head :math:`r` from the density field

    .. math::
        r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.
    """
    with timed_stage('density_solve'):
        solver.density_solver.solve()
    with timed_stage('rho_integral'):
        solver.rho_integrator.solve()
        solver.fields.baroc_head_3d *= -physical_constants['rho0_inv']
    with timed_stage('int_pg_solve'):
        solver.int_pg_calculator.solve()


@PETSc.Log.EventDecorator("thetis.extend_function_to_3d")
def extend_function_to_3d(func, mesh_extruded):
    """
    Returns a 3D view of a 2D :class:`Function` on the extruded domain.

    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
    assert fs.mesh().geometric_dimension() == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, firedrake.VectorElement):
        # vector function space
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0,
                                        dim=2, vector=True)
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
    func_extended.source = func
    return func_extended


class ExtrudedFunction(Function):
    """
    A 2D :class:`Function` that provides a 3D view on the extruded domain.

    The 3D function can be accessed as `ExtrudedFunction.view_3d`.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    def __init__(self, *args, mesh_3d=None, **kwargs):
        """
        Create a 2D :class:`Function` with a 3D view on extruded mesh.

        :arg mesh_3d: Extruded 3D mesh where the function will be extended to.
        """
        # create the 2d function
        super().__init__(*args, **kwargs)

        if mesh_3d is not None:
            self.view_3d = extend_function_to_3d(self, mesh_3d)


class SubdomainProjector(object):
    """
    Projector that projects the restriction of an expression to the specified subdomain.
    """
    def __init__(self, v, v_out, subdomain_id, solver_parameters=None, constant_jacobian=True):

        if not isinstance(v, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only project UFL expression or Functions not '%s'" % type(v))

        self.v = v
        self.v_out = v_out

        V = v_out.function_space()

        p = TestFunction(V)
        q = TrialFunction(V)

        a = inner(p, q)*dx
        L = inner(p, v)*dx(subdomain_id)

        problem = LinearVariationalProblem(a, L, v_out,
                                           constant_jacobian=constant_jacobian)

        if solver_parameters is None:
            solver_parameters = {}

        solver_parameters.setdefault("ksp_type", "cg")

        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.SubdomainProjector.project")
    def project(self):
        """
        Apply the projection.
        """
        self.solver.solve()


@PETSc.Log.EventDecorator("thetis.compute_elem_height")
def compute_elem_height(zcoord, output):
    """
    Computes the element height on an extruded mesh.

    :arg zcoord: field that contains the z coordinates of the mesh
    :type zcoord: :class:`Function`
    :arg output: field where element height is stored
    :type output: :class:`Function`
    """
    fs_in = zcoord.function_space()
    fs_out = output.function_space()

    iterate = op2.ALL

    # NOTE height maybe <0 if mesh was extruded like that
    kernel = op2.Kernel("""
        void my_kernel(double *func, double *zcoord) {
            for ( int d = 0; d < %(nodes)d/2; d++ ) {
                for ( int c = 0; c < %(func_dim)d; c++ ) {
                    double dz = fabs(zcoord[%(func_dim)d*(2*d+1) + c] - zcoord[%(func_dim)d*2*d + c]);
                    func[%(output_dim)d*2*d + c] = dz;
                    func[%(output_dim)d*(2*d+1) + c] = dz;
                }
            }
        }""" % {'nodes': zcoord.cell_node_map().arity,
                'func_dim': zcoord.function_space().value_size,
                'output_dim': output.function_space().value_size},
        'my_kernel')
    op2.par_loop(
        kernel, fs_out.mesh().cell_set,
        output.dat(op2.WRITE, fs_out.cell_node_map()),
        zcoord.dat(op2.READ, fs_in.cell_node_map()),
        iteration_region=iterate)

    return output


@no_annotations
@PETSc.Log.EventDecorator("thetis.get_horizontal_elem_size_2d")
def get_horizontal_elem_size_2d(sol2d):
    """
    Computes horizontal element size from the 2D mesh

    :arg sol2d: 2D :class:`Function` where result is stored
    """
    p1_2d = sol2d.function_space()
    mesh = p1_2d.mesh()
    test = TestFunction(p1_2d)
    tri = TrialFunction(p1_2d)
    a = inner(test, tri) * dx
    l = inner(test, sqrt(CellVolume(mesh))) * dx
    sp = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    }
    solve(a == l, sol2d, solver_parameters=sp)


@PETSc.Log.EventDecorator("thetis.get_facet_areas")
def get_facet_areas(mesh):
    """
    Compute area of each facet of `mesh`. The facet areas are stored as a HDiv trace field.

    NOTES:
      * In the 2D case, this gives edge lengths.
      * The plus sign is arbitrary and could equally well be chosen as minus.
    """
    HDivTrace = FunctionSpace(mesh, "HDiv Trace", 0)
    v, u = TestFunction(HDivTrace), TrialFunction(HDivTrace)
    facet_areas = Function(HDivTrace, name="Facet areas")
    mass_term = v('+')*u('+')*dS + v*u*ds
    rhs = v('+')*FacetArea(mesh)*dS + v*FacetArea(mesh)*ds
    sp = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
    solve(mass_term == rhs, facet_areas, solver_parameters=sp)
    return facet_areas


@PETSc.Log.EventDecorator("thetis.get_minimum_angles_2d")
def get_minimum_angles_2d(mesh2d):
    """
    Compute the minimum angle in each element of a triangular mesh, `mesh2d`, using the
    cosine rule. The minimum angles are outputted as a P0 field.
    """
    try:
        assert mesh2d.topological_dimension() == 2
        assert mesh2d.ufl_cell() == ufl.triangle
    except AssertionError:
        raise NotImplementedError("Minimum angle only currently implemented for triangles.")
    edge_lengths = get_facet_areas(mesh2d)
    min_angles = Function(FunctionSpace(mesh2d, "DG", 0))
    edge_cell_node_map = edge_lengths.function_space().cell_node_map()
    min_angle_cell_node_map = min_angles.function_space().cell_node_map()

    kernel = op2.Kernel("""
        void minimum_angle_kernel(double *edges, double *angle) {
            for (int i=0; i<%(nodes)d; i++) {
                  double min_edge = edges[0];
                  int min_index = 0;

                  for (int j=1; j<3; j++){
                    if (edges[j] < min_edge) {
                      min_edge = edges[j];
                      min_index = j;
                    }
                  }

                  double numerator = 0.0;
                  double denominator = 2.0;

                  for (int j=0; j<3; j++){
                    if (j == min_index) {
                      numerator -= edges[j]*edges[j];
                    } else {
                      numerator += edges[j]*edges[j];
                      denominator *= edges[j];
                    }
                  }
                  angle[0] = acos(numerator/denominator);
            }
        }""" % {"nodes": edge_cell_node_map.arity}, "minimum_angle_kernel")
    op2.par_loop(kernel, mesh2d.cell_set,
                 edge_lengths.dat(op2.READ, edge_cell_node_map),
                 min_angles.dat(op2.RW, min_angle_cell_node_map))
    return min_angles


@PETSc.Log.EventDecorator("thetis.get_cell_widths_2d")
def get_cell_widths_2d(mesh2d):
    """
    Compute widths of mesh elements in each coordinate direction as the maximum distance
    between components of vertex coordinates.
    """
    try:
        assert mesh2d.topological_dimension() == 2
        assert mesh2d.ufl_cell() == ufl.triangle
    except AssertionError:
        raise NotImplementedError("Cell widths only currently implemented for triangles.")
    cell_widths = Function(VectorFunctionSpace(mesh2d, "DG", 0)).assign(numpy.finfo(0.0).min)
    coords_cell_node_map = mesh2d.coordinates.function_space().cell_node_map()
    widths_cell_node_map = cell_widths.function_space().cell_node_map()
    kernel = op2.Kernel("""
        void cell_width_kernel(double *coords, double *widths) {
            for (int i=0; i<%(nodes)d; i++) {
                  widths[0] = fmax(widths[0], fabs(coords[2*i] - coords[(2*i+2)%%6]));
                  widths[1] = fmax(widths[1], fabs(coords[2*i+1] - coords[(2*i+3)%%6]));
            }
        }""" % {"nodes": coords_cell_node_map.arity}, "cell_width_kernel")
    op2.par_loop(kernel, mesh2d.cell_set,
                 mesh2d.coordinates.dat(op2.READ, coords_cell_node_map),
                 cell_widths.dat(op2.MAX, widths_cell_node_map))
    return cell_widths


@PETSc.Log.EventDecorator("thetis.anisotropic_cell_size")
def anisotropic_cell_size(mesh):
    """
    Measure of cell size for anisotropic meshes, as described in
    Micheletti et al. (2003).

    This is used in the SUPG formulation for the 2D tracer model.

    Micheletti, Perotto and Picasso (2003). Stabilized finite
    elements on anisotropic meshes: a priori error estimates for
    the advection-diffusion and the Stokes problems. SIAM Journal
    on Numerical Analysis 41.3: 1131-1162.
    """

    JTJ = dot(Jacobian(mesh).T, Jacobian(mesh))

    # based on https://github.com/michalhabera/dolfiny/blob/master/dolfiny/invariants.py
    I1 = JTJ[0, 0] + JTJ[1, 1]
    D = (JTJ[0, 0] - JTJ[1, 1])**2 + 4 * JTJ[0, 1] * JTJ[1, 0]
    D += numpy.finfo(float).eps

    # Compute smallest singular value of J
    P0 = FunctionSpace(mesh, "DG", 0)
    cell_size = Function(P0, name="Smallest singular value")
    cell_size.interpolate(sqrt((I1-sqrt(D))/2.))
    return cell_size


def beta_plane_coriolis_params(latitude):
    r"""
    Computes beta plane parameters :math:`f_0,\beta` based on latitude

    :arg float latitude: latitude in degrees
    :return: f_0, beta
    :rtype: float
    """
    omega = 7.2921150e-5  # rad/s Earth rotation rate
    r = 6371.e3  # Earth radius
    # Coriolis parameter f = 2 Omega sin(alpha)
    # Beta plane approximation f_beta = f_0 + Beta y
    # f_0 = 2 Omega sin(alpha_0)
    # Beta = df/dy|_{alpha=alpha_0}
    #      = (df/dalpha*dalpha/dy)_{alpha=alpha_0}
    #      = 2 Omega cos(alpha_0) /R
    alpha_0 = 2*numpy.pi*latitude/360.0
    f_0 = 2*omega*numpy.sin(alpha_0)
    beta = 2*omega*numpy.cos(alpha_0)/r
    return f_0, beta


def beta_plane_coriolis_function(latitude, out_function, y_offset=0.0):
    """
    Interpolates beta plane Coriolis function to a field

    :arg float latitude: latitude in degrees
    :arg out_function: :class:`Function` where to interpolate
    :kwarg float y_offset: offset (y - y_0) used in Beta-plane approximation.
        A constant in mesh coordinates.
    """
    # NOTE assumes that mesh y coordinate spans [-L_y, L_y]
    f0, beta = beta_plane_coriolis_params(latitude)
    coords = SpatialCoordinate(out_function.function_space().mesh())
    out_function.interpolate(f0 + beta * (coords[1] - y_offset))


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    .. math::
        \text{jump}(\mathbf{u}, \mathbf{n}) = (\mathbf{u}^+ \mathbf{n}^+) +
        (\mathbf{u}^- \mathbf{n}^-)

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator :math:`ufl.jump` which represents div(u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


def compute_boundary_length(mesh2d):
    """
    Computes the length of the boundary segments in given 2d mesh
    """
    p1 = get_functionspace(mesh2d, 'CG', 1)
    boundary_markers = sorted(mesh2d.exterior_facets.unique_markers)
    boundary_len = OrderedDict()
    for i in boundary_markers:
        ds_restricted = ds(int(i))
        one_func = Function(p1).assign(1.0)
        boundary_len[i] = assemble(one_func * ds_restricted)
    return boundary_len


def print_function_value_range(f, comm=COMM_WORLD, name=None, prefix=None,
                               format='2.3g'):
    """
    Prints the min/max DOF values of a function.

    .. code-block:: python

        print_function_value_range(f, name='myfunc', prefix='Initial')

    Prints `Initial myfunc 0.00 .. 0.00`.

    :kwarg comm: MPI communicator to use for the reduction
    :kwarg name: Optional function name. By default uses `f.name()`
    :kwarg prefix: Optional prefix for the output string
    :kwarg format: Value formatting string
    """
    if name is None:
        name = f.name()
    f_min = comm.allreduce(f.dat.data.min(), MPI.MIN)
    f_max = comm.allreduce(f.dat.data.max(), MPI.MAX)
    pre = prefix + ' ' if prefix is not None else ''
    bound_str_list = [f'{{:{format}}}'.format(v) for v in [f_min, f_max]]
    if numpy.allclose(f_min, f_max):
        print_output(f'{pre}{name}: {bound_str_list[0]}')
    else:
        print_output(f'{pre}{name}: {bound_str_list[0]} .. {bound_str_list[1]}')


@PETSc.Log.EventDecorator("thetis.select_and_move_detectors")
def select_and_move_detectors(mesh, detector_locations, detector_names=None,
                              maximum_distance=0.):
    """Select those detectors that are within the domain and/or move them to
    the nearest cell centre within the domain

    :arg mesh: Defines the domain in which detectors are to be located
    :arg detector_locations: List of x, y locations
    :arg detector_names: List of detector names (optional). If provided, a list
       of selected locations and a list of selected detector names are returned,
       otherwise only a list of selected locations is returned
    :arg maximum_distance: Detectors whose initial locations is outside the domain,
      but for which the nearest cell centre is within the specified distance, are
      moved to this location. By default a maximum distance of 0.0 is used, i.e
      no detectors are moved.
    """
    # auxilary function to test whether we can interpolate it in the given locations
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)

    P0 = FunctionSpace(mesh, "DG", 0)
    VP0 = VectorFunctionSpace(mesh, "DG", 0)
    dist = Function(P0)
    loc_const = Constant(detector_locations[0])
    xy = SpatialCoordinate(mesh)
    p0xy = Function(VP0).interpolate(xy)

    # comparison operator that sorts on first entry first, etc.
    def min_lexsort(x, y, datatype):
        for xi, yi in zip(x, y):
            if xi < yi:
                return x
            elif yi < xi:
                return y
        # all entries the same:
        return x
    min_lexsort_op = MPI.Op.Create(min_lexsort, commute=False)

    def move_to_nearest_cell_center(location):
        loc_const.assign(location)
        dist.interpolate(dot(xy-loc_const, xy-loc_const))
        ind = dist.dat.data_ro.argmin()
        # smallest distance to a cell centre location on this process:
        local_loc = list(p0xy.dat.data_ro[ind])
        local_dist = numpy.sqrt(dist.dat.data_ro[ind])
        # select the smallest distance on all processes. If some distances are equal, pick a unique loc. based on lexsort
        global_dist_loc = mesh.comm.allreduce([local_dist]+local_loc, op=min_lexsort_op)
        return global_dist_loc[0], global_dist_loc[1:]

    accepted_locations = []
    accepted_names = []
    if detector_names is None:
        names = [None] * len(detector_locations)
    else:
        names = detector_names
    for location, name in zip(detector_locations, names):
        try:
            v(location)
        except PointNotInDomainError:
            moved_dist, location = move_to_nearest_cell_center(location)
            if moved_dist > maximum_distance:
                continue
        accepted_locations.append(location)
        accepted_names.append(name)

    min_lexsort_op.Free()

    if detector_names is None:
        return accepted_locations
    else:
        return accepted_locations, accepted_names


class DepthExpression:
    r"""
    Construct expression for depth depending on options

    If `not use_nonlinear_equations`, then the depth is simply the bathymetry:

  ..math::
        H = h

    Otherwise we include the free surface elevation:

  ..math::
        H = h + \eta

    and if `use_wetting_and_drying`, includes a bathymetry displacement term
    to ensure a positive depth (see Karna et al. 2011):

  ..math::
        H = h + f(h+\eta) + \eta

    where

  ..math::
        f(h+\eta) = (\sqrt{(h+\eta)^2 +\alpha^2} - (h+\eta))/2

    This introduces a wetting-drying parameter :math:`\alpha`, with dimensions
    of length. The value for :math:`\alpha` is specified by
    :attr:`.ModelOptions.wetting_and_drying_alpha`, in units of meters. The
    default value is 0.5, but the appropriate value is problem specific and
    should be set by the user.
    """

    def __init__(self, bathymetry_2d, use_nonlinear_equations=True,
                 use_wetting_and_drying=False, wetting_and_drying_alpha=0.5):
        self.bathymetry_2d = bathymetry_2d
        self.use_nonlinear_equations = use_nonlinear_equations
        self.use_wetting_and_drying = use_wetting_and_drying
        self.wetting_and_drying_alpha = wetting_and_drying_alpha

    def wd_bathymetry_displacement(self, eta):
        """
        Returns wetting and drying bathymetry displacement as described in:
        Karna et al.,  2011.
        :arg eta: current elevation as UFL expression
        """
        if self.use_wetting_and_drying:
            H = self.bathymetry_2d + eta
            return 0.5 * (sqrt(H ** 2 + self.wetting_and_drying_alpha ** 2) - H)
        else:
            return 0

    def get_total_depth(self, eta):
        """
        Returns total water column depth based on options
        :arg eta: current elevation as UFL expression
        """
        if self.use_nonlinear_equations:
            total_h = self.bathymetry_2d + eta + self.wd_bathymetry_displacement(eta)
        else:
            total_h = self.bathymetry_2d
        return total_h


class DepthIntegratedPoissonSolver(object):
    r"""
    Construct solvers for Poisson equation and updating velocities

    Poisson equation is related to 2d NH SWE system

    Non-hydrostatic pressure :math:`q` is obtained from the generic form of equation

    .. math::
        \nabla \cdot \nabla q^{n+1/2} + A \cdot \nabla q^{n+1/2} + B q^{n+1/2} + C = 0

    The parameter terms A, B and C are defined as

    .. math::
        A = \frac{\nabla (\eta^* - d)}{H^*}
        B = \nabla A - \frac{4}{(H^*)^2}
        C = -2 \frac{\rho_0}{\Delta t} ( \nabla \cdot \bar{\textbf{u}}^* + 2 \frac{\bar{w} - w_b}{H^*} )

    where the :math:`H = \eta + d` denotes the water depth
    and the superscript star symbol represents the intermediate level of terms.
    """
    @PETSc.Log.EventDecorator("thetis.DepthIntegratedPoissonSolver.__init__")
    def __init__(self, q_2d, uv_2d, w_2d, elev_2d, depth, dt, bnd_functions=None, solver_parameters=None):
        if solver_parameters is None:
            solver_parameters = {'snes_type': 'ksponly',
                                 'ksp_type': 'preonly',
                                 'mat_type': 'aij',
                                 'pc_type': 'lu'}
        rho_0 = physical_constants['rho0']
        self.q_2d = q_2d
        self.uv_2d = uv_2d
        self.w_2d = w_2d
        self.elev_2d = elev_2d
        self.depth = depth
        self.dt = dt
        self.bnd_functions = bnd_functions

        fs_q = self.q_2d.function_space()
        test_q = TestFunction(fs_q)
        normal = FacetNormal(fs_q.mesh())
        boundary_markers = fs_q.mesh().exterior_facets.unique_markers

        bath_2d = self.depth.bathymetry_2d
        h_star = self.depth.get_total_depth(self.elev_2d)
        w_b = -dot(self.uv_2d, grad(bath_2d))  # TODO account for bed movement

        # weak form of `div(grad(q^{n+1/2}))`
        f = -dot(grad(self.q_2d), grad(test_q))*dx
        # weak form of `dot(grad(self.elev_2d - bath_2d)/h_star, grad(q^{n+1/2}))`
        grad_hori = grad(self.elev_2d - bath_2d)
        f += dot(grad_hori/h_star, grad(self.q_2d))*test_q*dx
        # weak form of `q^{n+1/2}*div(grad(self.elev_2d - bath_2d))/h_star`
        f += -dot(grad(self.q_2d*test_q/h_star), grad_hori)*dx
        # weak form of `-q^{n+1/2}*(dot(grad(self.elev_2d - bath_2d), grad(h_star)) + 4)/h_star**2`
        f += -(dot(grad_hori, grad(h_star)) + 4.)/h_star**2*self.q_2d*test_q*dx
        # weak form of `-2.*rho_0/self.dt*(div(self.uv_2d) + (self.w_2d - w_b)/(0.5*h_star))`
        const = 2.*rho_0/self.dt
        f += const*(dot(grad(test_q), self.uv_2d)*dx - 2.*(self.w_2d - w_b)/h_star*test_q*dx)

        # boundary conditions
        bcs = []
        for bnd_marker in boundary_markers:
            func = self.bnd_functions['shallow_water'].get(bnd_marker)
            ds_bnd = ds(int(bnd_marker))
            if func is not None:  # e.g. inlet flow, TODO be more precise
                bc = DirichletBC(fs_q, 0., int(bnd_marker))
                bcs.append(bc)
                f += self.q_2d*test_q/h_star*dot(grad_hori, normal)*ds_bnd
                f += -const*dot(self.uv_2d, normal)*test_q*ds_bnd

        prob_q = NonlinearVariationalProblem(f, self.q_2d)
        self.solver_q = NonlinearVariationalSolver(
            prob_q,
            solver_parameters=solver_parameters,
            options_prefix='poisson_solver'
        )
        # horizontal velocity updater
        fs_u = self.uv_2d.function_space()
        tri_u = TrialFunction(fs_u)
        test_u = TestFunction(fs_u)
        a_u = inner(tri_u, test_u)*dx
        l_u = dot(self.uv_2d - 0.5*self.dt/rho_0*(grad(self.q_2d) + grad_hori/h_star*self.q_2d), test_u)*dx
        prob_u = LinearVariationalProblem(a_u, l_u, self.uv_2d)
        self.solver_u = LinearVariationalSolver(prob_u)
        # vertical velocity updater
        fs_w = self.w_2d.function_space()
        tri_w = TrialFunction(fs_w)
        test_w = TestFunction(fs_w)
        a_w = inner(tri_w, test_w)*dx
        l_w = dot(self.w_2d + self.dt/rho_0*(self.q_2d/h_star), test_w)*dx
        prob_w = LinearVariationalProblem(a_w, l_w, self.w_2d)
        sp = {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
        self.solver_w = LinearVariationalSolver(prob_w, solver_parameters=sp)

    @PETSc.Log.EventDecorator("thetis.DepthIntegratedPoissonSolver.solve")
    def solve(self, solve_w=True):
        # solve non-hydrostatic pressure q
        self.solver_q.solve()
        # update horizontal velocity uv_2d
        self.solver_u.solve()
        # update vertical velocity w_2d
        if solve_w:
            self.solver_w.solve()


@PETSc.Log.EventDecorator("thetis.form2indicator")
def form2indicator(F):
    r"""
    Deduce the cell-wise contributions to a functional.

    Given a 0-form, multiply the integrand of each of its integrals by a
    :math:`\mathbb P0` test function and reassemble to give an element-wise error
    indicator.

    Note that a 0-form does not contain any :class:`firedrake.ufl_expr.TestFunction`\s
    or :class:`firedrake.ufl_expr.TrialFunction`\s.

    :arg F: the 0-form
    :return: the corresponding error indicator field

    Modified code based on
    https://github.com/pyroteus/goalie/blob/main/goalie/error_estimation.py
    """
    if len(F.arguments()) > 0:
        raise ValueError("Input form should be 0-form")
    mesh = F.ufl_domain()
    P0 = FunctionSpace(mesh, "DG", 0)
    p0test = TestFunction(P0)
    h = ufl.CellVolume(mesh)

    rhs = 0
    for integral in F.integrals_by_type("exterior_facet"):
        dsi = ds(integral.subdomain_id())
        rhs += h * p0test * integral.integrand() * dsi
    for integral in F.integrals_by_type("interior_facet"):
        dSi = dS(integral.subdomain_id())
        rhs += h("+") * p0test("+") * integral.integrand() * dSi
        rhs += h("-") * p0test("-") * integral.integrand() * dSi
    for integral in F.integrals_by_type("cell"):
        dxi = dx(integral.subdomain_id())
        rhs += h * p0test * integral.integrand() * dxi

    assert rhs != 0
    indicator = Function(P0)
    solve(
        TrialFunction(P0) * p0test * dx == rhs,
        indicator,
        solver_parameters={
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        },
    )
    return indicator
