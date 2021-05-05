"""
Module for three dimensional baroclinic solver
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import momentum_eq
from . import tracer_eq
from . import turbulence
from . import coupled_timeintegrator
import thetis.limiter as limiter
import time as time_mod
import numpy as np
from mpi4py import MPI
from . import exporter
import weakref
from .field_defs import field_metadata
from .options import ModelOptions3d
from . import callback
from .log import *
from collections import OrderedDict


class FlowSolver(FrozenClass):
    """
    Main object for 3D solver

    **Example**

    Create 2D mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create a 3D model with 6 uniform levels, and set some options
    (see :class:`.ModelOptions3d`)

    .. code-block:: python

        solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers=6)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'SSPRK22'
        options.timestepper_options.use_automatic_timestep = False
        options.solve_salinity = False
        options.solve_temperature = False
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(2.0*exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options=None, extrude_options=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: 2D :class:`Function`
        :arg int n_layers: Number of layers in the vertical direction.
            Elements are distributed uniformly over the vertical.
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions3d` instance
        """
        self._initialized = False

        self.bathymetry_cg_2d = bathymetry_2d

        self.mesh2d = mesh2d
        """2D :class`Mesh`"""
        if extrude_options is None:
            extrude_options = {}
        self.mesh = extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, **extrude_options)
        """3D :class`Mesh`"""
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh.boundary_len = bnd_len

        self.dt = None
        """Time step"""
        self.dt_2d = None
        """Time of the 2D solver"""
        self.M_modesplit = None
        """Mode split ratio (int)"""

        # override default options
        self.options = ModelOptions3d()
        """
        Dictionary of all options. A :class:`.ModelOptions3d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {},
                              'temp': {},
                              }

        self.callbacks = callback.CallbackManager()
        """
        :class:`.CallbackManager` object that stores all callbacks
        """

        self.fields = FieldDict()
        """
        :class:`.FieldDict` that holds all functions needed by the solver
        object
        """

        self.function_spaces = AttrDict()
        """
        :class:`.AttrDict` that holds all function spaces needed by the
        solver object
        """

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self._simulation_continued = False
        self._isfrozen = True

    def compute_dx_factor(self):
        """
        Computes normalized distance between nodes in the horizontal direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        if self.options.element_family in ['rt-dg', 'bdm-dg']:
            # velocity space is essentially p+1
            p = self.options.polynomial_degree + 1
        # assuming DG basis functions on triangles
        l_r = p**2/3.0 + 7.0/6.0*p + 1.0
        factor = 0.5*0.25/l_r
        return factor

    def compute_dz_factor(self):
        """
        Computes a normalized distance between nodes in the vertical direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        # assuming DG basis functions in an interval
        l_r = 1.0/max(p, 1)
        factor = 0.5*0.25/l_r
        return factor

    def compute_dt_2d(self, u_scale):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :arg u_scale: User provided maximum advective velocity scale
        :type u_scale: float or :class:`Constant`
        """
        csize = self.fields.h_elem_size_2d
        bath = self.fields.bathymetry_2d
        fs = bath.function_space()
        bath_pos = Function(fs, name='bathymetry')
        bath_pos.assign(bath)
        min_depth = 0.05
        bath_pos.dat.data[bath_pos.dat.data < min_depth] = min_depth
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solution = Function(fs)
        g = physical_constants['g_grav']
        u = (sqrt(g * bath_pos) + u_scale)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        dt = float(solution.dat.data.min())
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        dt *= self.compute_dx_factor()
        return dt

    def compute_dt_h_advection(self, u_scale):
        r"""
        Computes maximum explicit time step for horizontal advection

        .. math :: \Delta t = \frac{\Delta x}{U_{scale}}

        where :math:`U_{scale}` is estimated horizontal advective velocity.

        :arg u_scale: User provided maximum horizontal velocity scale
        :type u_scale: float or :class:`Constant`
        """
        u = u_scale
        if isinstance(u_scale, Constant):
            u = u_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/10.0/(self.options.polynomial_degree + 1)*min_dx/u
        min_dx *= self.compute_dx_factor()
        dt = min_dx/u
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_v_advection(self, w_scale):
        r"""
        Computes maximum explicit time step for vertical advection

        .. math :: \Delta t = \frac{\Delta z}{W_{scale}}

        where :math:`W_{scale}` is estimated vertical advective velocity.

        :arg w_scale: User provided maximum vertical velocity scale
        :type w_scale: float or :class:`Constant`
        """
        w = w_scale
        if isinstance(w_scale, Constant):
            w = w_scale.dat.data[0]
        min_dz = self.fields.v_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/1.5/(self.options.polynomial_degree + 1)*min_dz/w
        min_dz *= self.compute_dz_factor()
        dt = min_dz/w
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_diffusion(self, nu_scale):
        r"""
        Computes maximum explicit time step for horizontal diffusion.

        .. math :: \Delta t = \alpha \frac{(\Delta x)^2}{\nu_{scale}}

        where :math:`\nu_{scale}` is estimated diffusivity scale.
        """
        nu = nu_scale
        if isinstance(nu_scale, Constant):
            nu = nu_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        factor = 2.0
        if self.options.element_family == 'bdm-dg':
            factor = 1.8
        if self.options.timestepper_type == 'LeapFrog':
            factor *= 0.6
        min_dx *= factor*self.compute_dx_factor()
        dt = (min_dx)**2/nu
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        nnodes = self.function_spaces.P1_2d.dim()
        ntriangles = int(self.function_spaces.P1DG_2d.dim()/3)
        nlayers = self.mesh.topology.layers - 1
        nprisms = ntriangles*nlayers
        dofs_per_elem = len(self.function_spaces.H.finat_element.entity_dofs())
        ntracer_dofs = dofs_per_elem*nprisms
        min_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.min(), MPI.MIN)
        max_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.max(), MPI.MAX)
        min_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.min(), MPI.MIN)
        max_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.max(), MPI.MAX)

        print_output('2D mesh: {:} nodes, {:} triangles'.format(nnodes, ntriangles))
        print_output('3D mesh: {:} layers, {:} prisms'.format(nlayers, nprisms))
        print_output('Horizontal element size: {:.2f} ... {:.2f} m'.format(min_h_size, max_h_size))
        print_output('Vertical element size: {:.3f} ... {:.3f} m'.format(min_v_size, max_v_size))
        print_output('Element family: {:}, degree: {:}'.format(self.options.element_family, self.options.polynomial_degree))
        print_output('Number of tracer DOFs: {:}'.format(ntracer_dofs))
        print_output('Number of cores: {:}'.format(self.comm.size))
        print_output('Tracer DOFs per core: ~{:.1f}'.format(float(ntracer_dofs)/self.comm.size))

    def set_time_step(self):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions3d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions3d.timestep`.

        Once the time step is determined, will adjust it to be an integer
        fraction of export interval ``options.simulation_export_time``.
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep')
                              and self.options.timestepper_options.use_automatic_timestep)

        cfl2d = self.timestepper.cfl_coeff_2d
        cfl3d = self.timestepper.cfl_coeff_3d
        max_dt_swe = self.compute_dt_2d(self.options.horizontal_velocity_scale)
        max_dt_hadv = self.compute_dt_h_advection(self.options.horizontal_velocity_scale)
        max_dt_vadv = self.compute_dt_v_advection(self.options.vertical_velocity_scale)
        max_dt_diff = self.compute_dt_diffusion(self.options.horizontal_viscosity_scale)
        print_output('  - dt 2d swe: {:}'.format(max_dt_swe))
        print_output('  - dt h. advection: {:}'.format(max_dt_hadv))
        print_output('  - dt v. advection: {:}'.format(max_dt_vadv))
        print_output('  - dt viscosity: {:}'.format(max_dt_diff))
        max_dt_2d = cfl2d*max_dt_swe
        max_dt_3d = cfl3d*min(max_dt_hadv, max_dt_vadv, max_dt_diff)
        print_output('  - CFL adjusted dt: 2D: {:} 3D: {:}'.format(max_dt_2d, max_dt_3d))
        if not automatic_timestep:
            print_output('  - User defined dt: 2D: {:} 3D: {:}'.format(self.options.timestep_2d, self.options.timestep))
        self.dt = self.options.timestep
        self.dt_2d = self.options.timestep_2d
        if automatic_timestep:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            assert self.options.timestep_2d is not None
            assert self.options.timestep_2d > 0.0

        if self.dt_mode == 'split':
            if automatic_timestep:
                self.dt = max_dt_3d
                self.dt_2d = max_dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        elif self.dt_mode == '2d':
            if automatic_timestep:
                self.dt = min(max_dt_2d, max_dt_3d)
            self.dt_2d = self.dt
            self.M_modesplit = 1
        elif self.dt_mode == '3d':
            if automatic_timestep:
                self.dt = max_dt_3d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        print_output('  - chosen dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        # fit dt to export time
        m_exp = int(np.ceil(self.options.simulation_export_time/self.dt))
        self.dt = float(self.options.simulation_export_time)/m_exp
        if self.dt_mode == 'split':
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            self.dt_2d = self.dt

        print_output('  - adjusted dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        print_output('dt = {0:f}'.format(self.dt))
        if self.dt_mode == 'split':
            print_output('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
        sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0 = get_functionspace(self.mesh, 'DG', 0, 'DG', 0, name='P0')
        self.function_spaces.P1 = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1')
        self.function_spaces.P1v = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1v', vector=True)
        self.function_spaces.P1DG = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DG')
        self.function_spaces.P1DGv = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DGv', vector=True)

        # function spaces for (u,v) and w
        if self.options.element_family in ['rt-dg', 'bdm-dg']:
            h_family_prefix = self.options.element_family.split('-')[0].upper()
            h_family_suffix = {'triangle': 'F', 'quadrilateral': 'CF'}
            h_cell = self.mesh2d.ufl_cell().cellname()
            hfam = h_family_prefix + h_family_suffix[h_cell]
            h_degree = self.options.polynomial_degree + 1
            self.function_spaces.U = get_functionspace(self.mesh, hfam, h_degree, 'DG', self.options.polynomial_degree, name='U', hdiv=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'CG', self.options.polynomial_degree+1, name='W', hdiv=True)
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, hfam, h_degree, name='U_2d')
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='U', vector=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='W', vector=True)
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='H')
        self.function_spaces.turb_space = self.function_spaces.P0

        # 2D spaces
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P1v_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1v_2d', vector=True)
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DGv_2d', vector=True)
        self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d], name='V_2d')

        # define function spaces for baroclinic head and internal pressure gradient
        if self.options.use_quadratic_pressure:
            self.function_spaces.P2DGxP2 = get_functionspace(self.mesh, 'DG', 2, 'CG', 2, name='P2DGxP2')
            self.function_spaces.P2DG_2d = get_functionspace(self.mesh2d, 'DG', 2, name='P2DG_2d')
            self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
            self.function_spaces.H_bhead_2d = self.function_spaces.P2DG_2d
            if self.options.element_family == 'dg-dg':
                self.function_spaces.P2DGxP1DGv = get_functionspace(self.mesh, 'DG', 2, 'DG', 1, name='P2DGxP1DGv', vector=True, dim=2)
                self.function_spaces.U_int_pg = self.function_spaces.P2DGxP1DGv
            else:
                self.function_spaces.U_int_pg = self.function_spaces.U
        else:
            self.function_spaces.P1DGxP2 = get_functionspace(self.mesh, 'DG', 1, 'CG', 2, name='P1DGxP2')
            self.function_spaces.H_bhead = self.function_spaces.P1DGxP2
            self.function_spaces.H_bhead_2d = self.function_spaces.P1DG_2d
            self.function_spaces.U_int_pg = self.function_spaces.U

        self._isfrozen = True

    def create_fields(self):
        """
        Creates all fields
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # mesh velocity etc fields must be in the same space as 3D coordinates
        coord_is_dg = element_continuity(self.mesh2d.coordinates.function_space().ufl_element()).horizontal == 'dg'
        if coord_is_dg:
            coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
            coord_fs_2d = self.function_spaces.P1DG_2d
        else:
            coord_fs = self.function_spaces.P1
            coord_fs_2d = self.function_spaces.P1_2d

        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        # elevation field seen by the 3D mode, different from elev_2d
        self.fields.elev_domain_2d = ExtrudedFunction(self.function_spaces.H_2d, mesh_3d=self.mesh)
        self.fields.elev_cg_2d = ExtrudedFunction(coord_fs_2d, mesh_3d=self.mesh)
        self.fields.uv_3d = Function(self.function_spaces.U)
        self.fields.bathymetry_2d = ExtrudedFunction(coord_fs_2d, mesh_3d=self.mesh)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(coord_fs)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(coord_fs)
        self.fields.uv_dav_3d = Function(self.function_spaces.U)
        self.fields.uv_dav_2d = Function(self.function_spaces.U_2d)
        self.fields.split_residual_2d = Function(self.function_spaces.U_2d)
        self.fields.w_3d = Function(self.function_spaces.W)
        self.fields.hcc_metric_3d = Function(self.function_spaces.P1DG, name='mesh consistency')
        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d = Function(coord_fs)
        if self.options.solve_salinity:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solve_temperature:
            self.fields.temp_3d = Function(self.function_spaces.H, name='Temperature')
        if self.options.use_baroclinic_formulation:
            if self.options.use_quadratic_density:
                self.fields.density_3d = Function(self.function_spaces.P2DGxP2, name='Density')
            else:
                self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.H_bhead)
            self.fields.int_pg_3d = Function(self.function_spaces.U_int_pg, name='int_pg_3d')
        if self.options.coriolis_frequency is not None:
            if isinstance(self.options.coriolis_frequency, Constant):
                self.fields.coriolis_3d = self.options.coriolis_frequency
            else:

                self.fields.coriolis_3d = extend_function_to_3d(self.options.coriolis_frequency, self.mesh)
        if self.options.wind_stress is not None:
            if isinstance(self.options.wind_stress, Function):
                assert self.options.wind_stress.function_space().mesh().geometric_dimension() == 3, \
                    'wind stress field must be a 3D function'
                self.fields.wind_stress_3d = self.options.wind_stress
            elif isinstance(self.options.wind_stress, Constant):
                self.fields.wind_stress_3d = self.options.wind_stress
            else:
                raise Exception('Unsupported wind stress type: {:}'.format(type(self.options.wind_stress)))
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_3d(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.use_smagorinsky_viscosity:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
            self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.H)
        else:
            self.tracer_limiter = None
        if (self.options.use_limiter_for_velocity
                and self.options.polynomial_degree > 0
                and self.options.element_family == 'dg-dg'):
            self.uv_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.U)
        else:
            self.uv_limiter = None
        if self.options.use_turbulence:
            if self.options.turbulence_model_type == 'gls':
                # NOTE tke and psi should be in H as tracers ??
                self.fields.tke_3d = Function(self.function_spaces.turb_space)
                self.fields.psi_3d = Function(self.function_spaces.turb_space)
                # NOTE other turb. quantities should share the same nodes ??
                self.fields.eps_3d = Function(self.function_spaces.turb_space)
                self.fields.len_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence.GenericLengthScaleModel(
                    weakref.proxy(self),
                    self.fields.tke_3d,
                    self.fields.psi_3d,
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.len_3d,
                    self.fields.eps_3d,
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            elif self.options.turbulence_model_type == 'pacanowski':
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence.PacanowskiPhilanderModel(
                    weakref.proxy(self),
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            else:
                raise Exception('Unsupported turbulence model: {:}'.format(self.options.turbulence_model))
        else:
            self.turbulence_model = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = SumFunction()
        self.tot_h_visc.add(self.options.horizontal_viscosity)
        self.tot_h_visc.add(self.fields.get('smag_visc_3d'))
        self.tot_v_visc = SumFunction()
        self.tot_v_visc.add(self.options.vertical_viscosity)
        self.tot_v_visc.add(self.fields.get('eddy_visc_3d'))
        self.tot_h_diff = SumFunction()
        self.tot_h_diff.add(self.options.horizontal_diffusivity)
        self.tot_v_diff = SumFunction()
        self.tot_v_diff.add(self.options.vertical_diffusivity)
        self.tot_v_diff.add(self.fields.get('eddy_diff_3d'))

        self._isfrozen = True

    def create_equations(self):
        """
        Creates all dynamic equations and time integrators
        """
        if 'uv_3d' not in self.fields:
            self.create_fields()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        self.depth = DepthExpression(self.fields.bathymetry_2d,
                                     use_nonlinear_equations=self.options.use_nonlinear_equations)

        self.eq_sw = shallowwater_eq.ModeSplit2DEquations(
            self.fields.solution_2d.function_space(),
            self.depth, self.options)

        expl_bottom_friction = self.options.use_bottom_friction and not self.options.use_implicit_vertical_diffusion
        self.eq_momentum = momentum_eq.MomentumEquation(self.fields.uv_3d.function_space(),
                                                        bathymetry=self.fields.bathymetry_2d.view_3d,
                                                        v_elem_size=self.fields.v_elem_size_3d,
                                                        h_elem_size=self.fields.h_elem_size_3d,
                                                        use_nonlinear_equations=self.options.use_nonlinear_equations,
                                                        use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                        use_bottom_friction=expl_bottom_friction,
                                                        sipg_factor=self.options.sipg_factor,
                                                        sipg_factor_vertical=self.options.sipg_factor_vertical)
        if self.options.use_implicit_vertical_diffusion:
            self.eq_vertmomentum = momentum_eq.MomentumEquation(self.fields.uv_3d.function_space(),
                                                                bathymetry=self.fields.bathymetry_2d.view_3d,
                                                                v_elem_size=self.fields.v_elem_size_3d,
                                                                h_elem_size=self.fields.h_elem_size_3d,
                                                                use_nonlinear_equations=False,
                                                                use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                                use_bottom_friction=self.options.use_bottom_friction,
                                                                sipg_factor=self.options.sipg_factor,
                                                                sipg_factor_vertical=self.options.sipg_factor_vertical)
        if self.options.solve_salinity:
            self.eq_salt = tracer_eq.TracerEquation(self.fields.salt_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_2d.view_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg',
                                                    sipg_factor=self.options.sipg_factor_tracer,
                                                    sipg_factor_vertical=self.options.sipg_factor_vertical_tracer)
            if self.options.use_implicit_vertical_diffusion:
                self.eq_salt_vdff = tracer_eq.TracerEquation(self.fields.salt_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_2d.view_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                             sipg_factor=self.options.sipg_factor_tracer,
                                                             sipg_factor_vertical=self.options.sipg_factor_vertical_tracer)

        if self.options.solve_temperature:
            self.eq_temp = tracer_eq.TracerEquation(self.fields.temp_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_2d.view_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg',
                                                    sipg_factor=self.options.sipg_factor_tracer,
                                                    sipg_factor_vertical=self.options.sipg_factor_vertical_tracer)
            if self.options.use_implicit_vertical_diffusion:
                self.eq_temp_vdff = tracer_eq.TracerEquation(self.fields.temp_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_2d.view_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                             sipg_factor=self.options.sipg_factor_tracer,
                                                             sipg_factor_vertical=self.options.sipg_factor_vertical_tracer)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solve_salinity:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.solve_temperature:
            self.eq_temp.bnd_functions = self.bnd_functions['temp']
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_eq.TracerEquation(self.fields.tke_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_2d.view_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                           sipg_factor=self.options.sipg_factor_turb,
                                                           sipg_factor_vertical=self.options.sipg_factor_vertical_turb)
                self.eq_psi_adv = tracer_eq.TracerEquation(self.fields.psi_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_2d.view_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                           sipg_factor=self.options.sipg_factor_turb,
                                                           sipg_factor_vertical=self.options.sipg_factor_vertical_turb)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.TKEEquation(self.fields.tke_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_2d.view_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence.PsiEquation(self.fields.psi_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_2d.view_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)

        # ----- Time integrators
        self.dt_mode = '3d'  # 'split'|'2d'|'3d' use constant 2d/3d dt, or split
        if self.options.timestepper_type == 'LeapFrog':
            self.timestepper = coupled_timeintegrator.CoupledLeapFrogAM3(weakref.proxy(self))
        elif self.options.timestepper_type == 'SSPRK22':
            self.timestepper = coupled_timeintegrator.CoupledTwoStageRK(weakref.proxy(self))
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))

        # ----- File exporters
        # create export_managers and store in a list
        self.exporters = OrderedDict()
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.output_directory,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.output_directory, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        # ----- Operators
        tot_uv_3d = self.fields.uv_3d + self.fields.uv_dav_3d
        self.w_solver = VerticalVelocitySolver(self.fields.w_3d,
                                               tot_uv_3d,
                                               self.fields.bathymetry_2d.view_3d,
                                               self.eq_momentum.bnd_functions)
        self.uv_averager = VerticalIntegrator(self.fields.uv_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=Constant((0.0, 0.0, 0.0)),
                                              average=True,
                                              bathymetry=self.fields.bathymetry_2d.view_3d,
                                              elevation=self.fields.elev_cg_2d.view_3d)
        if self.options.use_baroclinic_formulation:
            if self.options.solve_salinity:
                s = self.fields.salt_3d
            else:
                s = self.options.constant_salinity
            if self.options.solve_temperature:
                t = self.fields.temp_3d
            else:
                t = self.options.constant_temperature
            if self.options.equation_of_state_type == 'linear':
                eos_options = self.options.equation_of_state_options
                self.equation_of_state = LinearEquationOfState(eos_options.rho_ref,
                                                               eos_options.alpha,
                                                               eos_options.beta,
                                                               eos_options.th_ref,
                                                               eos_options.s_ref)
            else:
                self.equation_of_state = JackettEquationOfState()
            if self.options.use_quadratic_density:
                self.density_solver = DensitySolverWeak(s, t, self.fields.density_3d,
                                                        self.equation_of_state)
            else:
                self.density_solver = DensitySolver(s, t, self.fields.density_3d,
                                                    self.equation_of_state)
            self.rho_integrator = VerticalIntegrator(self.fields.density_3d,
                                                     self.fields.baroc_head_3d,
                                                     bottom_to_top=False,
                                                     average=False,
                                                     bathymetry=self.fields.bathymetry_2d.view_3d,
                                                     elevation=self.fields.elev_cg_2d.view_3d)
            self.int_pg_calculator = momentum_eq.InternalPressureGradientCalculator(
                self.fields, self.fields.bathymetry_2d.view_3d,
                self.bnd_functions['momentum'],
                internal_pg_scalar=self.options.internal_pg_scalar,
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
        self.copy_uv_dav_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                                                           elem_height=self.fields.v_elem_size_3d)
        self.copy_uv_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                       elem_height=self.fields.v_elem_size_3d)
        self.mesh_updater = ALEMeshUpdater(self)

        if self.options.use_smagorinsky_viscosity:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_coefficient, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff,
                                                                weak_form=True)

        # ----- set initial values
        self.fields.bathymetry_2d.project(self.bathymetry_cg_2d)
        self.mesh_updater.initialize()
        self.compute_mesh_stats()
        self.set_time_step()
        self.timestepper.set_dt(self.dt, self.dt_2d)
        # compute maximal diffusivity for explicit schemes
        degree_h, degree_v = self.function_spaces.H.ufl_element().degree()
        max_diff_alpha = 1.0/60.0/max((degree_h*(degree_h + 1)), 1.0)  # FIXME depends on element type and order
        self.fields.max_h_diff.assign(max_diff_alpha/self.dt * self.fields.h_elem_size_3d**2)
        d = self.fields.max_h_diff.dat.data
        print_output('max h diff {:} - {:}'.format(d.min(), d.max()))

        self.next_export_t = self.simulation_time + self.options.simulation_export_time
        self._initialized = True
        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, salt=None, temp=None,
                                  uv_2d=None, uv_3d=None, tke=None, psi=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg salt: Initial condition for salinity field
        :type salt: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg temp: Initial condition for temperature field
        :type temp: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_2d: Initial condition for depth averaged velocity
        :type uv_2d: vector valued 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_3d: Initial condition for horizontal velocity
        :type uv_3d: vector valued 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg tke: Initial condition for turbulent kinetic energy field
        :type tke: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg psi: Initial condition for turbulence generic length scale field
        :type psi: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.create_equations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            self.fields.uv_dav_2d.project(uv_2d)
            if uv_3d is None:
                ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                   elem_height=self.fields.v_elem_size_3d).solve()
        if uv_3d is not None:
            self.fields.uv_3d.project(uv_3d)
        if salt is not None and self.options.solve_salinity:
            self.fields.salt_3d.project(salt)
        if temp is not None and self.options.solve_temperature:
            self.fields.temp_3d.project(temp)
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if tke is not None:
                self.fields.tke_3d.project(tke)
            if psi is not None:
                self.fields.psi_3d.project(psi)
            self.turbulence_model.initialize()

        if self.options.use_ale_moving_mesh:
            self.timestepper._update_3d_elevation()
            self.timestepper._update_moving_mesh()
        self.timestepper.initialize()
        # update all diagnostic variables
        self.timestepper._update_all_dependencies(self.simulation_time,
                                                  do_2d_coupling=False,
                                                  do_vert_diffusion=False,
                                                  do_ale_update=True,
                                                  do_stab_params=True,
                                                  do_turbulence=False)
        if self.options.use_turbulence:
            self.turbulence_model.initialize()

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg str eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export', index=self.i_export)
        # set uv to total uv instead of deviation from depth average
        # TODO find a cleaner way of doing this ...
        self.fields.uv_3d += self.fields.uv_dav_3d
        for e in self.exporters.values():
            e.export()
        # restore uv_3d
        self.fields.uv_3d -= self.fields.uv_dav_3d

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format.  The required
        state variables are: elev_2d, uv_2d, uv_3d, salt_3d, temp_3d, tke_3d,
        psi_3d

        Currently hdf5 field import only works for the same number of MPI
        processes.

        :arg int i_export: export index to load
        :kwarg string outputdir: (optional) directory where files are read from.
            By default ``options.output_directory``.
        :kwarg float t: simulation time. Overrides the time stamp stored in the
            hdf5 files.
        :kwarg int iteration: Overrides the iteration count in the hdf5 files.
        """
        if not self._initialized:
            self.create_equations()
        if outputdir is None:
            outputdir = self.options.output_directory
        self._simulation_continued = True
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d', 'uv_3d',
                        'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        e.exporters['uv_3d'].load(i_export, self.fields.uv_3d)
        # NOTE remove mean from uv_3d
        self.timestepper._remove_depth_average_from_uv_3d()
        salt = temp = tke = psi = None
        if self.options.solve_salinity:
            salt = self.fields.salt_3d
            e.exporters['salt_3d'].load(i_export, salt)
        if self.options.solve_temperature:
            temp = self.fields.temp_3d
            e.exporters['temp_3d'].load(i_export, temp)
        if self.options.use_turbulence:
            if 'tke_3d' in self.fields:
                tke = self.fields.tke_3d
                e.exporters['tke_3d'].load(i_export, tke)
            if 'psi_3d' in self.fields:
                psi = self.fields.psi_3d
                e.exporters['psi_3d'].load(i_export, psi)
        self.assign_initial_conditions(elev=self.fields.elev_2d,
                                       uv_2d=self.fields.uv_2d,
                                       uv_3d=self.fields.uv_3d,
                                       salt=salt, temp=temp,
                                       tke=tke, psi=psi,
                                       )

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.simulation_export_time
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.output_directory
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.simulation_export_time
        for e in self.exporters.values():
            e.set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.elev_2d)
        norm_u = norm(self.fields.uv_3d)

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def _print_field(self, field):
        """
        Prints min/max values of a field for debugging.

        :arg field: a :class:`Function` or a field string, e.g. 'salt_3d'
        """
        if isinstance(field, str):
            _field = self.fields[field]
        else:
            _field = field
        minval = float(_field.dat.data.min())
        minval = self.comm.allreduce(minval, op=MPI.MIN)
        maxval = float(_field.dat.data.max())
        maxval = self.comm.allreduce(maxval, op=MPI.MAX)
        print_output('    {:}: {:.4f} {:.4f}'.format(_field.name(), minval, maxval))

    def print_state_debug(self):
        """
        Print min/max values of prognostic/diagnostic fields for debugging.
        """

        field_list = [
            'elev_2d', 'uv_2d', 'elev_domain_2d', 'elev_cg_2d', 'uv_3d',
            'w_3d', 'uv_dav_3d', 'w_mesh_3d',
            'salt_3d', 'temp_3d', 'density_3d',
            'baroc_head_3d', 'int_pg_3d',
            'psi_3d', 'eps_3d', 'eddy_visc_3d',
            'shear_freq_3d', 'buoy_freq_3d',
            'coriolis_2d', 'coriolis_3d',
            'wind_stress_3d',
        ]
        print_output('{:06} T={:10.2f}'.format(self.iteration, self.simulation_time))
        for fieldname in field_list:
            if (fieldname in self.fields
                    and isinstance(self.fields[fieldname], Function)):
                self._print_field(self.fields[fieldname])
        self.comm.barrier()

    def iterate(self, update_forcings=None, update_forcings3d=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 2D system (if any).
        :kwarg update_forcings_3d: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 3D equations (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        if not self._initialized:
            self.create_equations()

        self.options.check_salinity_conservation &= self.options.solve_salinity
        self.options.check_salinity_overshoot &= self.options.solve_salinity
        self.options.check_temperature_conservation &= self.options.solve_temperature
        self.options.check_temperature_overshoot &= self.options.solve_temperature
        self.options.check_volume_conservation_3d &= self.options.use_ale_moving_mesh
        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.element_family == 'dg-dg'

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.perf_counter()

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_volume_conservation_3d:
            c = callback.VolumeConservation3DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_conservation:
            c = callback.TracerMassConservationCallback('salt_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_overshoot:
            c = callback.TracerOvershootCallBack('salt_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_conservation:
            c = callback.TracerMassConservationCallback('temp_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_overshoot:
            c = callback.TracerOvershootCallBack('temp_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self._simulation_continued:
            # set all callbacks to append mode
            for m in self.callbacks:
                for k in self.callbacks[m]:
                    self.callbacks[m][k].set_write_mode('append')

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            self.timestepper.advance(self.simulation_time,
                                     update_forcings, update_forcings3d)

            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= self.next_export_t - t_epsilon:
                self.i_export += 1
                self.next_export_t += self.options.simulation_export_time

                cputime = time_mod.perf_counter() - cputimestamp
                cputimestamp = time_mod.perf_counter()
                self.print_state(cputime)

                self.export()
                if export_func is not None:
                    export_func()
