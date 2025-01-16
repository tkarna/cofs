"""
Utility solvers and calculators for 3D hydrostatic ocean model
"""
from .utility import *
from abc import ABC, abstractmethod
import numpy


__all__ = [
    "VerticalVelocitySolver",
    "VerticalIntegrator",
    "DensitySolver",
    "DensitySolverWeak",
    "VelocityMagnitudeSolver",
    "Mesh3DConsistencyCalculator",
    "ExpandFunctionTo3d",
    "SubFunctionExtractor",
    "ALEMeshUpdater",
    "SmagorinskyViscosity",
    "EquationOfState",
    "JackettEquationOfState",
    "LinearEquationOfState",
    "get_horizontal_elem_size_3d",
]


class VerticalVelocitySolver(object):
    r"""
    Computes vertical velocity diagnostically from the continuity equation

    Vertical velocity is obtained from the continuity equation

    .. math::
        \frac{\partial w}{\partial z} = -\nabla_h \cdot \textbf{u}
        :label: continuity_eq_3d

    and the bottom impermeability condition (:math:`h` denotes the bathymetry)

    .. math::
        \textbf{n}_h \cdot \textbf{u} + w n_z &= 0 \quad \forall \mathbf{x} \in \Gamma_{b} \\
        \Leftrightarrow w &= -\nabla_h h \cdot \mathbf{u} \quad \forall \mathbf{x} \in \Gamma_{b}

    :math:`w` can be solved with the weak form

    .. math::
        \int_{\Gamma_s} w n_z \varphi dS
        + \int_{\mathcal{I}_h} \text{avg}(w) \text{jump}(\varphi n_z) dS
        - \int_{\Omega} w \frac{\partial \varphi}{\partial z} dx
        = \\
        \int_{\Omega} \mathbf{u} \cdot \nabla_h \varphi dx
        - \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{avg}(\mathbf{u}) \cdot \text{jump}(\varphi \mathbf{n}_h) dS
        - \int_{\Gamma_s} \mathbf{u} \cdot \varphi \mathbf{n}_h dS

    where the :math:`\Gamma_b` terms vanish due to the bottom impermeability
    condition.
    """
    @PETSc.Log.EventDecorator("thetis.VerticalVelocitySolver.__init__")
    def __init__(self, solution, uv, bathymetry, boundary_funcs={},
                 solver_parameters=None):
        """
        :arg solution: w :class:`Function`
        :arg uv: horizontal velocity :class:`Function`
        :arg bathymetry: bathymetry :class:`Function`
        :kwarg dict boundary_funcs: boundary conditions used in the 3D momentum
            equation. Provides external values of uv (if any).
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')
        solver_parameters.setdefault('sub_pc_factor_shift_type', 'inblocks')

        fs = solution.function_space()
        mesh = fs.mesh()
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        normal = FacetNormal(mesh)

        # define measures with a reasonable quadrature degree
        p, q = fs.ufl_element().degree()
        self.quad_degree = (2*p, 2*q)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)

        # NOTE weak dw/dz
        a = tri[2]*test[2]*normal[2]*ds_surf + \
            avg(tri[2])*jump(test[2], normal[2])*dS_h - Dx(test[2], 2)*tri[2]*self.dx

        # NOTE weak div(uv)
        uv_star = avg(uv)
        # NOTE in the case of mimetic uv the div must be taken over all components
        l_v_facet = (uv_star[0]*jump(test[2], normal[0])
                     + uv_star[1]*jump(test[2], normal[1])
                     + uv_star[2]*jump(test[2], normal[2]))*self.dS_v
        l_h_facet = (uv_star[0]*jump(test[2], normal[0])
                     + uv_star[1]*jump(test[2], normal[1])
                     + uv_star[2]*jump(test[2], normal[2]))*self.dS_h
        l_surf = (uv[0]*normal[0]
                  + uv[1]*normal[1] + uv[2]*normal[2])*test[2]*self.ds_surf
        l_vol = inner(uv, nabla_grad(test[2]))*self.dx
        l = l_vol - l_v_facet - l_h_facet - l_surf
        for bnd_marker in sorted(mesh.exterior_facets.unique_markers):
            funcs = boundary_funcs.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
            if funcs is None:
                # assume land boundary
                continue
            else:
                # use symmetry condition
                l += -(uv[0]*normal[0] + uv[1]*normal[1])*test[2]*ds_bnd

        # NOTE For ALE mesh constant_jacobian should be False
        # however the difference is very small as A is nearly independent of
        # mesh stretching: only the normals vary in time
        self.prob = LinearVariationalProblem(a, l, solution,
                                             constant_jacobian=True)
        self.solver = LinearVariationalSolver(self.prob,
                                              solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.VerticalVelocitySolver.solve")
    def solve(self):
        """Compute w"""
        self.solver.solve()


class VerticalIntegrator(object):
    """
    Computes vertical integral (or average) of a field.
    """
    @PETSc.Log.EventDecorator("thetis.VerticalIntegrator.__init__")
    def __init__(self, input, output, bottom_to_top=True,
                 bnd_value=Constant(0.0), average=False,
                 bathymetry=None, elevation=None, solver_parameters=None):
        """
        :arg input: 3D field to integrate
        :arg output: 3D field where the integral is stored
        :kwarg bottom_to_top: Defines the integration direction: If True integration is performed along the z axis, from bottom surface to top surface.
        :kwarg bnd_value: Value of the integral at the bottom (top) boundary if bottom_to_top is True (False)
        :kwarg average: If True computes the vertical average instead. Requires bathymetry and elevation fields
        :kwarg bathymetry: 3D field defining the bathymetry
        :kwarg elevation: 3D field defining the free surface elevation
        :kwarg dict solver_parameters: PETSc solver options
        """
        self.output = output
        space = output.function_space()
        mesh = space.mesh()
        e_continuity = element_continuity(space.ufl_element())
        vertical_is_dg = e_continuity.vertical in ['dg', 'hdiv']

        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        if e_continuity.vertical != 'hdiv':
            solver_parameters.setdefault('ksp_type', 'preonly')
            solver_parameters.setdefault('pc_type', 'bjacobi')
            solver_parameters.setdefault('sub_ksp_type', 'preonly')
            solver_parameters.setdefault('sub_pc_type', 'ilu')

        tri = TrialFunction(space)
        phi = TestFunction(space)
        normal = FacetNormal(mesh)

        # define measures with a reasonable quadrature degree
        p, q = space.ufl_element().degree()
        p_in, q_in = input.function_space().ufl_element().degree()
        self.quad_degree = (p+p_in+1, q+q_in+1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

        if bottom_to_top:
            bnd_term = normal[2]*inner(bnd_value, phi)*self.ds_bottom
            mass_bnd_term = normal[2]*inner(tri, phi)*self.ds_surf
        else:
            bnd_term = normal[2]*inner(bnd_value, phi)*self.ds_surf
            mass_bnd_term = normal[2]*inner(tri, phi)*self.ds_bottom

        self.a = -inner(Dx(phi, 2), tri)*self.dx + mass_bnd_term
        if bottom_to_top:
            up_value = tri('+')
        else:
            up_value = tri('-')
        if vertical_is_dg:
            if len(input.ufl_shape) > 0:
                dim = input.ufl_shape[0]
                for i in range(dim):
                    self.a += up_value[i]*jump(phi[i], normal[2])*self.dS_h
            else:
                self.a += up_value*jump(phi, normal[2])*self.dS_h
        if average:
            source = input/(elevation + bathymetry)
        else:
            source = input
        self.l = inner(source, phi)*self.dx + bnd_term
        self.prob = LinearVariationalProblem(self.a, self.l, output, constant_jacobian=average)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.VerticalIntegrator.solve")
    def solve(self):
        """
        Computes the integral and stores it in the output field.
        """
        self.solver.solve()


class DensitySolver(object):
    r"""
    Computes density from salinity and temperature using the equation of state.

    Water density is defined as

    .. math::
        \rho = \rho'(T, S, p) + \rho_0

    This method computes the density anomaly :math:`\rho'`.

    Density is computed point-wise assuming that temperature, salinity and
    density are in the same function space.
    """
    @PETSc.Log.EventDecorator("thetis.DensitySolver.__init__")
    def __init__(self, salinity, temperature, density, eos_class):
        """
        :arg salinity: water salinity field
        :type salinity: :class:`Function`
        :arg temperature: water temperature field
        :type temperature: :class:`Function`
        :arg density: water density field
        :type density: :class:`Function`
        :arg eos_class: equation of state that defines water density
        :type eos_class: :class:`EquationOfState`
        """
        self.fs = density.function_space()
        self.eos = eos_class

        if isinstance(salinity, Function):
            assert self.fs == salinity.function_space()
        if isinstance(temperature, Function):
            assert self.fs == temperature.function_space()

        self.s = salinity
        self.t = temperature
        self.rho = density

    def _get_array(self, function):
        """Returns numpy data array from a :class:`Function`"""
        if isinstance(function, Function):
            assert self.fs == function.function_space()
            return function.dat.data[:]
        if isinstance(function, Constant):
            return float(function)
        # assume that function is a float
        return function

    @PETSc.Log.EventDecorator("thetis.DensitySolver.solve")
    def solve(self):
        """Compute density"""
        s = self._get_array(self.s)
        th = self._get_array(self.t)
        p = 0.0  # NOTE ignore pressure for now
        rho0 = self._get_array(physical_constants['rho0'])
        self.rho.dat.data[:] = self.eos.compute_rho(s, th, p, rho0)


class DensitySolverWeak(object):
    r"""
    Computes density from salinity and temperature using the equation of state.

    Water density is defined as

    .. math::
        \rho = \rho'(T, S, p) + \rho_0

    This method computes the density anomaly :math:`\rho'`.

    Density is computed in a weak sense by projecting the analytical expression
    on the density field.
    """
    @PETSc.Log.EventDecorator("thetis.DensitySolverWeak.__init__")
    def __init__(self, salinity, temperature, density, eos_class):
        """
        :arg salinity: water salinity field
        :type salinity: :class:`Function`
        :arg temperature: water temperature field
        :type temperature: :class:`Function`
        :arg density: water density field
        :type density: :class:`Function`
        :arg eos_class: equation of state that defines water density
        :type eos_class: :class:`EquationOfState`
        """
        self.fs = density.function_space()
        self.eos = eos_class

        assert isinstance(salinity, (Function, Constant))
        assert isinstance(temperature, (Function, Constant))

        self.s = salinity
        self.t = temperature
        self.density = density
        self.p = Constant(0.)
        rho0 = physical_constants['rho0']

        f = self.eos.eval(self.s, self.t, self.p, rho0)
        self.projector = Projector(f, self.density)

    def ensure_positive_salinity(self):
        """
        make sure salinity is not negative

        some EOS depend on sqrt(salt).
        """
        # FIXME this is really hacky and modifies the state variable
        # NOTE if salt field is P2 checking nodal values is not enough ..
        ix = self.s.dat.data < 0
        self.s.dat.data[ix] = 0.0

    @PETSc.Log.EventDecorator("thetis.DensitySolverWeak.solve")
    def solve(self):
        """Compute density"""
        self.ensure_positive_salinity()
        self.projector.project()


class VelocityMagnitudeSolver(object):
    """
    Computes magnitude of (u[0],u[1],w) and stores it in solution
    """
    @PETSc.Log.EventDecorator("thetis.VelocityMagnitudeSolver.__init__")
    def __init__(self, solution, u=None, w=None, min_val=1e-6,
                 solver_parameters=None):
        """
        :arg solution: scalar field for velocity magnitude scalar :class:`Function`
        :type solution: :class:`Function`
        :kwarg u: horizontal velocity
        :type u: :class:`Function`
        :kwarg w: vertical velocity
        :type w: :class:`Function`
        :kwarg float min_val: minimum value of magnitude. Minimum value of solution
            will be clipped to this value
        :kwarg dict solver_parameters: PETSc solver options


        If ``u`` is None computes magnitude of (0,0,w).

        If ``w`` is None computes magnitude of (u[0],u[1],0).
        """
        self.solution = solution
        self.min_val = min_val
        function_space = solution.function_space()
        test = TestFunction(function_space)
        tri = TrialFunction(function_space)

        a = test*tri*dx
        s = 0
        if u is not None:
            s += u[0]**2 + u[1]**2
        if w is not None:
            s += w**2
        l = test*sqrt(s)*dx
        self.prob = LinearVariationalProblem(a, l, solution)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.VelocityMagnitudeSolver.solve")
    def solve(self):
        """Compute the magnitude"""
        self.solver.solve()
        numpy.maximum(self.solution.dat.data, self.min_val, self.solution.dat.data)


class Mesh3DConsistencyCalculator(object):
    r"""
    Computes a hydrostatic consistency criterion metric on the 3D mesh.

    Let :math:`\Delta x` and :math:`\Delta z` denote the horizontal and vertical
    element sizes. The hydrostatic consistency criterion (HCC) can then be
    expressed as

    .. math::
        R = \frac{|\nabla h| \Delta x}{\Delta z} < 1

    where :math:`\nabla h` is the bathymetry gradient (or gradient of the
    internal horizontal facet).

    Violating the hydrostatic consistency criterion leads to internal pressure
    gradient errors.
    In practice one can violate the :math:`R < 1` condition without
    jeopardizing numerical stability; typically :math:`R < 5`.
    Mesh consistency can be improved by coarsening the vertical
    mesh, refining the horizontal mesh, or smoothing the bathymetry.

    For a 3D prism, let :math:`\delta z_t,\delta z_b` denote the maximal
    :math:`z` coordinate difference in the surface and bottom facets,
    respectively, and :math:`\Delta z` the height of the prism.
    We can then compute :math:`R` for the two facets as

    .. math::
        R_t &= \frac{\delta z_t}{\Delta z} \\
        R_b &= \frac{\delta z_b}{\Delta z}

    For a straight prism we have :math:`R = 0`, and :math:`R = 1` in
    the case where the highest bottom node is at the same level as the lowest
    surface node.
    """
    @PETSc.Log.EventDecorator("thetis.Mesh3DConsistencyCalculator.__init__")
    def __init__(self, solver_obj):
        """
        :arg solver_obj: :class:`FlowSolver` object
        """
        self.solver_obj = solver_obj
        self.output = self.solver_obj.fields.hcc_metric_3d
        self.z_coord = solver_obj.fields.z_coord_3d

        # create par loop for computing delta
        self.fs_3d = self.solver_obj.function_spaces.P1DG
        assert self.output.function_space() == self.fs_3d

        nodes = get_facet_mask(self.fs_3d, 'bottom')
        self.idx = op2.Global(len(nodes), nodes, dtype=numpy.int32, name='node_idx')
        self.kernel = op2.Kernel("""
            void my_kernel(double *output, double *z_field, int *idx) {
                // compute max delta z on top and bottom facets
                double z_top_max = -1e20;
                double z_top_min = 1e20;
                double z_bot_max = -1e20;
                double z_bot_min = 1e20;
                int i_top = 1;
                int i_bot = 0;
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    double z_top = z_field[idx[d] + i_top];
                    double z_bot = z_field[idx[d] + i_bot];
                    z_top_max = fmax(z_top, z_top_max);
                    z_top_min = fmin(z_top, z_top_min);
                    z_bot_max = fmax(z_bot, z_bot_max);
                    z_bot_min = fmin(z_bot, z_bot_min);
                }
                double delta_z_top = z_top_max - z_top_min;
                double delta_z_bot = z_bot_max - z_bot_min;
                // compute R ratio
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    double z_top = z_field[idx[d] + i_top];
                    double z_bot = z_field[idx[d] + i_bot];
                    double h = z_top - z_bot;
                    output[idx[d] + i_top] = delta_z_top/h;
                    output[idx[d] + i_bot] = delta_z_bot/h;
                }
            }""" % {'nodes': len(nodes)},
            'my_kernel')

    @PETSc.Log.EventDecorator("thetis.Mesh3DConsistencyCalculator.solve")
    def solve(self):
        """Compute the HCC metric"""
        op2.par_loop(self.kernel, self.solver_obj.mesh.cell_set,
                     self.output.dat(op2.WRITE, self.output.function_space().cell_node_map()),
                     self.z_coord.dat(op2.READ, self.z_coord.function_space().cell_node_map()),
                     self.idx(op2.READ),
                     iteration_region=op2.ALL)
        # compute global min/max
        r_min = self.output.dat.data.min()
        r_max = self.output.dat.data.max()
        r_min = self.solver_obj.comm.allreduce(r_min, op=MPI.MIN)
        r_max = self.solver_obj.comm.allreduce(r_max, op=MPI.MAX)
        print_output('HCC: {:} .. {:}'.format(r_min, r_max))


class ExpandFunctionTo3d(object):
    """
    Copy a 2D field to 3D

    Copies a field from 2D mesh to 3D mesh, assigning the same value over the
    vertical dimension. Horizontal function spaces must be the same.

    .. code-block:: python

        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)
        ex = ExpandFunctionTo3d(func2d, func3d)
        ex.solve()
    """
    @PETSc.Log.EventDecorator("thetis.ExpandFunctionTo3d.__init__")
    def __init__(self, input_2d, output_3d, elem_height=None):
        """
        :arg input_2d: 2D source field
        :type input_2d: :class:`Function`
        :arg output_3d: 3D target field
        :type output_3d: :class:`Function`
        :kwarg elem_height: scalar :class:`Function` in 3D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
        self.input_2d = input_2d
        self.output_3d = output_3d
        self.fs_2d = self.input_2d.function_space()
        self.fs_3d = self.output_3d.function_space()

        family_2d = self.fs_2d.ufl_element().family()
        base_element_3d = get_extruded_base_element(self.fs_3d.ufl_element())
        assert isinstance(base_element_3d, firedrake.TensorProductElement)
        family_3dh = base_element_3d.sub_elements[0].family()
        if family_2d != family_3dh:
            raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        self.do_hdiv_scaling = family_2d in ['Raviart-Thomas', 'RTCF', 'Brezzi-Douglas-Marini', 'BDMCF']
        if self.do_hdiv_scaling and elem_height is None:
            raise Exception('elem_height must be provided for HDiv spaces')

        self.iter_domain = op2.ALL

        # number of nodes in vertical direction
        n_vert_nodes = self.fs_3d.finat_element.space_dimension() / self.fs_2d.finat_element.space_dimension()

        nodes = get_facet_mask(self.fs_3d, 'bottom')
        self.idx = op2.Global(len(nodes), nodes, dtype=numpy.int32, name='node_idx')
        self.kernel = op2.Kernel("""
            void my_kernel(double *func, double *func2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func2d_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            func[%(func3d_dim)d*(idx[d]+e) + c] = func2d[%(func2d_dim)d*d + c];
                        }
                    }
                }
            }""" % {'nodes': self.fs_2d.finat_element.space_dimension(),
                    'func2d_dim': self.input_2d.function_space().value_size,
                    'func3d_dim': self.fs_3d.value_size,
                    'v_nodes': n_vert_nodes},
            'my_kernel')

        if self.do_hdiv_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_3d)
            tri = TrialFunction(self.fs_3d)
            a = inner(tri, test)*dx
            l = inner(self.output_3d, test)*elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_3d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.ExpandFunctionTo3d.solve")
    def solve(self):
        with timed_stage('copy_2d_to_3d'):
            # execute par loop
            op2.par_loop(
                self.kernel, self.fs_3d.mesh().cell_set,
                self.output_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
                self.input_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
                self.idx(op2.READ),
                iteration_region=self.iter_domain)

            if self.do_hdiv_scaling:
                self.rt_scale_solver.solve()


class SubFunctionExtractor(object):
    """
    Extract a 2D sub-function from a 3D function in an extruded mesh

    Given 2D and 3D functions,

    .. code-block:: python

        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)

    Get surface value:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='top', elem_facet='top')
        ex.solve()

    Get bottom value:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='bottom', elem_facet='bottom')
        ex.solve()

    Get value at the top of bottom element:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='bottom', elem_facet='top')
        ex.solve()
    """
    @PETSc.Log.EventDecorator("thetis.SubFunctionExtractor.__init__")
    def __init__(self, input_3d, output_2d,
                 boundary='top', elem_facet=None,
                 elem_height=None):
        """
        :arg input_3d: 3D source field
        :type input_3d: :class:`Function`
        :arg output_2d: 2D target field
        :type output_2d: :class:`Function`
        :kwarg str boundary: 'top'|'bottom'
            Defines whether to extract from the surface or bottom 3D elements
        :kwarg str elem_facet: 'top'|'bottom'|'average'
            Defines which facet of the 3D element is extracted. The 'average'
            computes mean of the top and bottom facets of the 3D element.
        :kwarg elem_height: scalar :class:`Function` in 2D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
        self.input_3d = input_3d
        self.output_2d = output_2d
        self.fs_3d = self.input_3d.function_space()
        self.fs_2d = self.output_2d.function_space()

        if elem_facet is None:
            # extract surface/bottom face by default
            elem_facet = boundary

        family_2d = self.fs_2d.ufl_element().family()
        base_element_3d = get_extruded_base_element(self.fs_3d.ufl_element())
        assert isinstance(base_element_3d, firedrake.TensorProductElement)
        family_3dh = base_element_3d.sub_elements[0].family()
        if family_2d != family_3dh:
            raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        self.do_hdiv_scaling = family_2d in ['Raviart-Thomas', 'RTCF', 'Brezzi-Douglas-Marini', 'BDMCF']
        if self.do_hdiv_scaling and elem_height is None:
            raise Exception('elem_height must be provided for HDiv spaces')

        assert elem_facet in ['top', 'bottom', 'average'], 'Unsupported elem_facet: {:}'.format(elem_facet)
        if elem_facet == 'average':
            nodes = numpy.hstack((get_facet_mask(self.fs_3d, 'bottom'),
                                  get_facet_mask(self.fs_3d, 'top')))
        else:
            nodes = get_facet_mask(self.fs_3d, elem_facet)
        if boundary == 'top':
            self.iter_domain = op2.ON_TOP
        elif boundary == 'bottom':
            self.iter_domain = op2.ON_BOTTOM

        out_nodes = self.fs_2d.finat_element.space_dimension()

        if elem_facet == 'average':
            assert (len(nodes) == 2*out_nodes)
        else:
            assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=numpy.int32, name='node_idx')
        if elem_facet == 'average':
            # compute average of top and bottom elem nodes
            self.kernel = op2.Kernel("""
                void my_kernel(double *func, double *func3d, int *idx) {
                    int nnodes = %(nodes)d;
                    for ( int d = 0; d < nnodes; d++ ) {
                        for ( int c = 0; c < %(func2d_dim)d; c++ ) {
                            func[%(func2d_dim)d*d + c] = 0.5*(func3d[%(func3d_dim)d*idx[d] + c] +
                                              func3d[%(func3d_dim)d*idx[d + nnodes] + c]);
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func2d_dim': self.output_2d.function_space().value_size,
                        'func3d_dim': self.fs_3d.value_size},
                'my_kernel')
        else:
            self.kernel = op2.Kernel("""
                void my_kernel(double *func, double *func3d, int *idx) {
                    for ( int d = 0; d < %(nodes)d; d++ ) {
                        for ( int c = 0; c < %(func2d_dim)d; c++ ) {
                            func[%(func2d_dim)d*d + c] = func3d[%(func3d_dim)d*idx[d] + c];
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func2d_dim': self.output_2d.function_space().value_size,
                        'func3d_dim': self.fs_3d.value_size},
                'my_kernel')

        if self.do_hdiv_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_2d)
            tri = TrialFunction(self.fs_2d)
            a = inner(tri, test)*dx
            l = inner(self.output_2d, test)/elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_2d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.SubFunctionExtractor.solve")
    def solve(self):
        with timed_stage('copy_3d_to_2d'):
            # execute par loop
            op2.par_loop(self.kernel, self.fs_3d.mesh().cell_set,
                         self.output_2d.dat(op2.WRITE, self.fs_2d.cell_node_map()),
                         self.input_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                         self.idx(op2.READ),
                         iteration_region=self.iter_domain)

            if self.do_hdiv_scaling:
                self.rt_scale_solver.solve()


class ALEMeshUpdater(object):
    """
    Class that handles vertically moving ALE mesh

    Mesh geometry is updated to match the elevation field
    (``solver.fields.elev_2d``). First the discontinuous elevation field is
    projected to continuous space, and this field is used to update the mesh
    coordinates.

    This class stores the reference coordinate field and keeps track of the
    updated mesh coordinates. It also provides a method for computing the mesh
    velocity from two adjacent elevation fields.
    """
    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.__init__")
    def __init__(self, solver):
        """
        :arg solver: :class:`FlowSolver` object
        """
        self.solver = solver
        self.fields = solver.fields
        if self.solver.options.use_ale_moving_mesh:
            # continous elevation
            self.elev_cg_2d = Function(self.solver.function_spaces.P1_2d,
                                       name='elev cg 2d')
            # w_mesh at surface
            self.w_mesh_surf_2d = Function(
                self.fields.bathymetry_2d.function_space(), name='w mesh surf 2d')
            # elevation in coordinate space
            self.proj_elev_to_cg_2d = Projector(self.fields.elev_2d,
                                                self.elev_cg_2d)
            self.proj_elev_cg_to_coords_2d = Projector(self.elev_cg_2d,
                                                       self.fields.elev_cg_2d)
        self.cp_v_elem_size_to_2d = SubFunctionExtractor(self.fields.v_elem_size_3d,
                                                         self.fields.v_elem_size_2d,
                                                         boundary='top', elem_facet='top')

        self.fs_3d = self.fields.z_coord_ref_3d.function_space()
        self.fs_2d = self.fields.elev_cg_2d.function_space()

        family_2d = self.fs_2d.ufl_element().family()
        base_element_3d = get_extruded_base_element(self.fs_3d.ufl_element())
        assert isinstance(base_element_3d, firedrake.TensorProductElement)
        family_3dh = base_element_3d.sub_elements[0].family()
        if family_2d != family_3dh:
            raise Exception('2D and 3D spaces do not match: "{0:s}" != "{1:s}"'.format(family_2d, family_3dh))

        # number of nodes in vertical direction
        n_vert_nodes = self.fs_3d.finat_element.space_dimension() / self.fs_2d.finat_element.space_dimension()

        nodes = get_facet_mask(self.fs_3d, 'bottom')
        self.idx = op2.Global(len(nodes), nodes, dtype=numpy.int32, name='node_idx')
        self.kernel_z_coord = op2.Kernel("""
            void my_kernel(double *z_coord_3d, double *z_ref_3d, double *elev_2d, double *bath_2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func2d_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            double eta = elev_2d[%(func2d_dim)d*d + c];
                            double bath = bath_2d[%(func2d_dim)d*d + c];
                            double z_ref = z_ref_3d[%(func3d_dim)d*(idx[d]+e) + c];
                            double new_z = eta*(z_ref + bath)/bath + z_ref;
                            z_coord_3d[%(func3d_dim)d*(idx[d]+e) + c] = new_z;
                        }
                    }
                }
            }""" % {'nodes': self.fs_2d.finat_element.space_dimension(),
                    'func2d_dim': self.fs_2d.value_size,
                    'func3d_dim': self.fs_3d.value_size,
                    'v_nodes': n_vert_nodes},
            'my_kernel')

        self.kernel_w_mesh = op2.Kernel("""
            void my_kernel(double *w_mesh_3d, double *z_ref_3d, double *w_mesh_surf_2d, double *bath_2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func2d_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            double w_mesh_surf = w_mesh_surf_2d[%(func2d_dim)d*d + c];
                            double bath = bath_2d[%(func2d_dim)d*d + c];
                            double z_ref = z_ref_3d[%(func3d_dim)d*(idx[d]+e) + c];
                            double new_w = w_mesh_surf * (z_ref + bath)/bath;
                            w_mesh_3d[%(func3d_dim)d*(idx[d]+e) + c] = new_w;
                        }
                    }
                }
            }""" % {'nodes': self.fs_2d.finat_element.space_dimension(),
                    'func2d_dim': self.fs_2d.value_size,
                    'func3d_dim': self.fs_3d.value_size,
                    'v_nodes': n_vert_nodes},
            'my_kernel')

    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.intialize")
    def initialize(self):
        """Set values for initial mesh (elevation at rest)"""
        get_zcoord_from_mesh(self.fields.z_coord_ref_3d)
        self.fields.z_coord_3d.assign(self.fields.z_coord_ref_3d)
        self.update_elem_height()

    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.update_elem_height")
    def update_elem_height(self):
        """Updates vertical element size fields"""
        compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
        self.cp_v_elem_size_to_2d.solve()

    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.compute_mesh_velocity_begin")
    def compute_mesh_velocity_begin(self):
        """Stores the current 2D elevation state as the "old" field"""
        assert self.solver.options.use_ale_moving_mesh
        self.proj_elev_to_cg_2d.project()
        self.proj_elev_cg_to_coords_2d.project()

    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.compute_mesh_velocity_finalize")
    def compute_mesh_velocity_finalize(self, c=1.0, w_mesh_surf_expr=None):
        """
        Computes mesh velocity from the elevation difference

        Stores the current 2D elevation state as the "new" field,
        and computes w_mesh using the given time step factor ``c``.
        """
        assert self.solver.options.use_ale_moving_mesh
        # compute w_mesh at surface
        if w_mesh_surf_expr is None:
            # default formulation
            # w_mesh_surf = (elev_new - elev_old)/dt/c
            self.w_mesh_surf_2d.assign(self.fields.elev_cg_2d)
            self.proj_elev_to_cg_2d.project()
            self.proj_elev_cg_to_coords_2d.project()
            self.w_mesh_surf_2d += -self.fields.elev_cg_2d
            self.w_mesh_surf_2d *= -1.0/self.solver.dt/c
        else:
            # user-defined formulation
            self.w_mesh_surf_2d.assign(w_mesh_surf_expr)
        op2.par_loop(
            self.kernel_w_mesh, self.fs_3d.mesh().cell_set,
            self.fields.w_mesh_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
            self.fields.z_coord_ref_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
            self.w_mesh_surf_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
            self.fields.bathymetry_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
            self.idx(op2.READ),
            iteration_region=op2.ALL
        )

    @PETSc.Log.EventDecorator("thetis.ALEMeshUpdater.update_mesh_coordinates")
    def update_mesh_coordinates(self):
        """
        Updates 3D mesh coordinates to match current elev_2d field

        elev_2d is first projected to continous space
        """
        assert self.solver.options.use_ale_moving_mesh
        self.proj_elev_to_cg_2d.project()
        self.proj_elev_cg_to_coords_2d.project()

        # compute new z coordinates -> self.fields.z_coord_3d
        op2.par_loop(
            self.kernel_z_coord, self.fs_3d.mesh().cell_set,
            self.fields.z_coord_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
            self.fields.z_coord_ref_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
            self.fields.elev_cg_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
            self.fields.bathymetry_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
            self.idx(op2.READ),
            iteration_region=op2.ALL
        )

        self.solver.mesh.coordinates.dat.data[:, 2] = self.fields.z_coord_3d.dat.data[:]
        self.update_elem_height()
        self.solver.mesh.clear_spatial_index()


class SmagorinskyViscosity(object):
    r"""
    Computes Smagorinsky subgrid scale horizontal viscosity

    This formulation is according to Ilicak et al. (2012) and
    Griffies and Hallberg (2000).

    .. math::
        \nu = (C_s \Delta x)^2 |S|

    with the deformation rate

    .. math::
        |S| &= \sqrt{D_T^2 + D_S^2} \\
        D_T &= \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y} \\
        D_S &= \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}

    :math:`\Delta x` is the horizontal element size and :math:`C_s` is the
    Smagorinsky coefficient.

    To match a certain mesh Reynolds number :math:`Re_h` set
    :math:`C_s = 1/\sqrt{Re_h}`.

    Ilicak et al. (2012). Spurious dianeutral mixing and the role of
    momentum closure. Ocean Modelling, 45-46(0):37-58.
    http://dx.doi.org/10.1016/j.ocemod.2011.10.003

    Griffies and Hallberg (2000). Biharmonic friction with a
    Smagorinsky-like viscosity for use in large-scale eddy-permitting
    ocean models. Monthly Weather Review, 128(8):2935-2946.
    http://dx.doi.org/10.1175/1520-0493(2000)128%3C2935:BFWASL%3E2.0.CO;2
    """
    @PETSc.Log.EventDecorator("thetis.SmagorinskyViscosity.__init__")
    def __init__(self, uv, output, c_s, h_elem_size, max_val, min_val=1e-10,
                 weak_form=True, solver_parameters=None):
        """
        :arg uv_3d: horizontal velocity
        :type uv_3d: 3D vector :class:`Function`
        :arg output: Smagorinsky viscosity field
        :type output: 3D scalar :class:`Function`
        :arg c_s: Smagorinsky coefficient
        :type c_s: float or :class:`Constant`
        :arg h_elem_size: field that defines the horizontal element size
        :type h_elem_size: 3D scalar :class:`Function` or :class:`Constant`
        :arg float max_val: Maximum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg float min_val: Minimum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg bool weak_form: Compute velocity shear by integrating by parts.
            Necessary for some function spaces (e.g. P0).
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        assert max_val.function_space() == output.function_space(), \
            'max_val function must belong to the same space as output'
        self.max_val = max_val
        self.min_val = min_val
        self.output = output
        self.weak_form = weak_form

        if self.weak_form:
            # solve grad(u) weakly
            mesh = output.function_space().mesh()
            fs_grad = get_functionspace(mesh, 'DP', 1, 'DP', 1, vector=True, dim=4)
            self.grad = Function(fs_grad, name='uv_grad')

            tri_grad = TrialFunction(fs_grad)
            test_grad = TestFunction(fs_grad)

            normal = FacetNormal(mesh)
            a = inner(tri_grad, test_grad)*dx

            rhs_terms = []
            for iuv in range(2):
                for ix in range(2):
                    i = 2*iuv + ix
                    vol_term = -inner(Dx(test_grad[i], ix), uv[iuv])*dx
                    int_term = inner(avg(uv[iuv]), jump(test_grad[i], normal[ix]))*dS_v
                    ext_term = inner(uv[iuv], test_grad[i]*normal[ix])*ds_v
                    rhs_terms.extend([vol_term, int_term, ext_term])
            l = sum(rhs_terms)
            prob = LinearVariationalProblem(a, l, self.grad)
            self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

            # rate of strain tensor
            d_t = self.grad[0] - self.grad[3]
            d_s = self.grad[1] + self.grad[2]
        else:
            # rate of strain tensor
            d_t = Dx(uv[0], 0) - Dx(uv[1], 1)
            d_s = Dx(uv[0], 1) + Dx(uv[1], 0)

        fs = output.function_space()
        tri = TrialFunction(fs)
        test = TestFunction(fs)

        nu = c_s**2*h_elem_size**2 * sqrt(d_t**2 + d_s**2)

        a = test*tri*dx
        l = test*nu*dx
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    @PETSc.Log.EventDecorator("thetis.SmagorinskyViscosity.solve")
    def solve(self):
        """Compute viscosity"""
        if self.weak_form:
            self.weak_grad_solver.solve()
        self.solver.solve()
        # remove negative values
        ix = self.output.dat.data < self.min_val
        self.output.dat.data[ix] = self.min_val

        # crop too large values
        ix = self.output.dat.data > self.max_val.dat.data
        self.output.dat.data[ix] = self.max_val.dat.data[ix]


class EquationOfState(ABC):
    """
    Base class of all equation of state objects
    """

    @abstractmethod
    def compute_rho(self, s, th, p, rho0=0.0):
        r"""
        Compute sea water density.

        :arg s: Salinity expressed on the Practical Salinity Scale 1978
        :type s: float or numpy.array
        :arg th: Potential temperature in Celsius, referenced to pressure
            p_r = 0 dbar.
        :type th: float or numpy.array
        :arg p: Pressure in decibars (1 dbar = 1e4 Pa)
        :type p: float or numpy.array
        :kwarg float rho0: Optional reference density. If provided computes
            :math:`\rho' = \rho(S, Th, p) - \rho_0`
        :return: water density
        :rtype: float or numpy.array

        All pressures are gauge pressures: they are the absolute pressures minus standard atmosperic
        pressure 10.1325 dbar.
        """
        pass

    @abstractmethod
    def eval(self, s, th, p, rho0=0.0):
        r"""
        Compute sea water density.
        """
        pass


class JackettEquationOfState(EquationOfState):
    r"""
    Equation of State according of Jackett et al. (2006) for computing sea
    water density.

    .. math ::
        \rho = \rho'(T, S, p) + \rho_0
        :label: equation_of_state

    :math:`\rho'(T, S, p)` is a nonlinear rational function.

    Jackett et al. (2006). Algorithms for Density, Potential Temperature,
    Conservative Temperature, and the Freezing Temperature of Seawater.
    Journal of Atmospheric and Oceanic Technology, 23(12):1709-1728.
    http://dx.doi.org/10.1175/JTECH1946.1
    """
    a = numpy.array([9.9984085444849347e2, 7.3471625860981584e0, -5.3211231792841769e-2,
                     3.6492439109814549e-4, 2.5880571023991390e0, -6.7168282786692355e-3,
                     1.9203202055760151e-3, 1.1798263740430364e-2, 9.8920219266399117e-8,
                     4.6996642771754730e-6, -2.5862187075154352e-8, -3.2921414007960662e-12])
    b = numpy.array([1.0, 7.2815210113327091e-3, -4.4787265461983921e-5, 3.3851002965802430e-7,
                     1.3651202389758572e-10, 1.7632126669040377e-3, -8.8066583251206474e-6,
                     -1.8832689434804897e-10, 5.7463776745432097e-6, 1.4716275472242334e-9,
                     6.7103246285651894e-6, -2.4461698007024582e-17, -9.1534417604289062e-18])

    def compute_rho(self, s, th, p, rho0=0.0):
        r"""
        Compute sea water density.

        :arg s: Salinity expressed on the Practical Salinity Scale 1978
        :type s: float or numpy.array
        :arg th: Potential temperature in Celsius, referenced to pressure
            p_r = 0 dbar.
        :type th: float or numpy.array
        :arg p: Pressure in decibars (1 dbar = 1e4 Pa)
        :type p: float or numpy.array
        :kwarg float rho0: Optional reference density. If provided computes
            :math:`\rho' = \rho(S, Th, p) - \rho_0`
        :return: water density
        :rtype: float or numpy.array

        All pressures are gauge pressures: they are the absolute pressures minus standard atmosperic
        pressure 10.1325 dbar.
        """
        s_pos = numpy.maximum(s, 0.0)  # ensure salinity is positive
        return self.eval(s_pos, th, p, rho0)

    def eval(self, s, th, p, rho0=0.0):
        a = self.a
        b = self.b
        pn = (a[0] + th*a[1] + th*th*a[2] + th*th*th*a[3] + s*a[4]
              + th*s*a[5] + s*s*a[6] + p*a[7] + p*th * th*a[8] + p*s*a[9]
              + p*p*a[10] + p*p*th*th * a[11])
        pd = (b[0] + th*b[1] + th*th*b[2] + th*th*th*b[3]
              + th*th*th*th*b[4] + s*b[5] + s*th*b[6] + s*th*th*th*b[7]
              + pow(s, 1.5)*b[8] + pow(s, 1.5)*th*th*b[9] + p*b[10]
              + p*p*th*th*th*b[11] + p*p*p*th*b[12])
        rho = pn/pd - rho0
        return rho


class LinearEquationOfState(EquationOfState):
    r"""
    Linear Equation of State for computing sea water density

    .. math::
        \rho = \rho_{ref} - \alpha (T - T_{ref}) + \beta (S - S_{ref})
    """
    def __init__(self, rho_ref, alpha, beta, th_ref, s_ref):
        """
        :arg float rho_ref: reference density
        :arg float alpha: thermal expansion coefficient
        :arg float beta: haline contraction coefficient
        :arg float th_ref: reference temperature
        :arg float s_ref: reference salinity
        """
        self.rho_ref = rho_ref
        self.alpha = alpha
        self.beta = beta
        self.th_ref = th_ref
        self.S_ref = s_ref

    def compute_rho(self, s, th, p, rho0=0.0):
        r"""
        Compute sea water density.

        :arg s: Salinity expressed on the Practical Salinity Scale 1978
        :type s: float or numpy.array
        :arg th: Potential temperature in Celsius
        :type th: float or numpy.array
        :arg p: Pressure in decibars (1 dbar = 1e4 Pa)
        :type p: float or numpy.array
        :kwarg float rho0: Optional reference density. If provided computes
            :math:`\rho' = \rho(S, Th, p) - \rho_0`
        :return: water density
        :rtype: float or numpy.array

        Pressure is ingored in this equation of state.
        """
        rho = (self.rho_ref - rho0
               - self.alpha*(th - self.th_ref)
               + self.beta*(s - self.S_ref))
        return rho

    def eval(self, s, th, p, rho0=0.0):
        return self.compute_rho(s, th, p, rho0)


@PETSc.Log.EventDecorator("thetis.get_horizontal_elem_size_3d")
def get_horizontal_elem_size_3d(sol2d, sol3d):
    """
    Computes horizontal element size from the 2D mesh, then copies it on a 3D
    field

    :arg sol2d: 2D :class:`Function` for the element size field
    :arg sol3d: 3D :class:`Function` for the element size field
    """
    get_horizontal_elem_size_2d(sol2d)
    ExpandFunctionTo3d(sol2d, sol3d).solve()
