"""
Module for 2D depth averaged solver
"""
from .utility import *
from . import shallowwater_eq
from . import timeintegrator
from . import rungekutta
from . import implicitexplicit
from . import coupled_timeintegrator_2d
from . import tracer_eq_2d
from . import sediment_eq_2d
from . import exner_eq
import weakref
import time as time_mod
from mpi4py import MPI
from . import exporter
from .turbines import TidalTurbineFarm, DiscreteTidalTurbineFarm
from .field_defs import field_metadata
from .options import ModelOptions2d
from . import callback
from .log import *
from collections import OrderedDict
import thetis.limiter as limiter
import numpy
import datetime


class FlowSolver2d(FrozenClass):
    """
    Main object for 2D depth averaged solver

    **Example**

    Create mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create solver object and set some options

    .. code-block:: python

        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.swe_timestepper_type = 'CrankNicolson'
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.__init__")
    def __init__(self, mesh2d, bathymetry_2d, options=None, keep_log=False):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions2d` instance
        :kwarg bool keep_log: append to an existing log file, or overwrite it?
        """
        self._initialized = False
        self.mesh2d = mesh2d
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len

        self.dt = None
        """Time step"""

        self.options = ModelOptions2d()
        """
        Dictionary of all options. A :class:`.ModelOptions2d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

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

        self.fields.bathymetry_2d = bathymetry_2d

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self.sediment_model = None
        """set up option for sediment model"""

        self.bnd_functions = {'shallow_water': {}, 'tracer': {}, 'sediment': {}}

        if 'tracer_2d' in field_metadata:
            field_metadata.pop('tracer_2d')
        self.solve_tracer = False
        self.keep_log = keep_log
        self._field_preproc_funcs = {}

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.compute_time_step")
    def compute_time_step(self, u_scale=Constant(0.0)):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :kwarg u_scale: User provided maximum advective velocity scale
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
        return solution

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.compute_mesh_stats")
    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        nnodes = self.function_spaces.P1_2d.dim()
        P1DG_2d = self.function_spaces.P1DG_2d
        nelem2d = int(P1DG_2d.dim()/P1DG_2d.ufl_cell().num_vertices())
        dofs_elev2d = self.function_spaces.H_2d.dim()
        dofs_u2d = self.function_spaces.U_2d.dim()
        dofs_tracer2d = self.function_spaces.Q_2d.dim()
        dofs_elev2d_core = int(dofs_elev2d/self.comm.size)
        dofs_tracer2d_core = int(dofs_tracer2d/self.comm.size)
        min_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.min(), MPI.MIN)
        max_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.max(), MPI.MAX)

        if not self.options.tracer_only:
            print_output(f'Element family: {self.options.element_family}, degree: {self.options.polynomial_degree}')
        if self.solve_tracer:
            print_output(f'Tracer element family: {self.options.tracer_element_family}, degree: 1')
        print_output(f'2D cell type: {self.mesh2d.ufl_cell()}')
        print_output(f'2D mesh: {nnodes} vertices, {nelem2d} elements')
        print_output(f'Horizontal element size: {min_h_size:.2f} ... {max_h_size:.2f} m')
        if not self.options.tracer_only:
            print_output(f'Number of 2D elevation DOFs: {dofs_elev2d}')
            print_output(f'Number of 2D velocity DOFs: {dofs_u2d}')
        if self.solve_tracer:
            print_output(f'Number of 2D tracer DOFs: {dofs_tracer2d}')
        print_output(f'Number of cores: {self.comm.size}')
        if not self.options.tracer_only:
            print_output(f'Elevation DOFs per core: ~{dofs_elev2d_core:.1f}')
        if self.solve_tracer:
            print_output(f'Tracer DOFs per core: ~{dofs_tracer2d_core:.1f}')

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.set_time_step")
    def set_time_step(self, alpha=0.05):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions2d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions2d.timestep`.

        :kwarg float alpha: CFL number scaling factor
        """
        automatic_timestep = False
        sed_options = self.options.sediment_model_options
        ts_options = (
            self.options.swe_timestepper_options, self.options.tracer_timestepper_options,
            sed_options.sediment_timestepper_options, sed_options.exner_timestepper_options,
            self.options.nh_model_options.free_surface_timestepper_options,
        )
        for options in ts_options:
            if hasattr(options, 'use_automatic_timestep') and options.use_automatic_timestep:
                automatic_timestep = True

        # TODO revisit math alpha is OBSOLETE
        if automatic_timestep:
            mesh2d_dt = self.compute_time_step(u_scale=self.options.horizontal_velocity_scale)
            dt = self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min())
            dt = self.comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        else:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            self.dt = self.options.timestep
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.set_wetting_and_drying_alpha")
    def set_wetting_and_drying_alpha(self):
        r"""
        Compute a wetting and drying parameter :math:`\alpha` which ensures positive water
        depth using the approximate method suggested by Karna et al. (2011).

        This method takes

      ..math::
            \alpha \approx \mid L_x \nabla h \mid,

        where :math:`L_x` is the horizontal length scale of the mesh elements at the wet-dry
        front and :math:`h` is the bathymetry profile.

        This expression is interpolated into :math:`\mathbb P1` space in order to remove noise. Note
        that we use the `interpolate` method, rather than the `project` method, in order to avoid
        introducing new extrema.

        NOTE: The minimum and maximum values at which to cap the alpha parameter may be specified via
        :attr:`ModelOptions2d.wetting_and_drying_alpha_min` and
        :attr:`ModelOptions2d.wetting_and_drying_alpha_max`.
        """
        if not self.options.use_wetting_and_drying:
            return
        if self.options.use_automatic_wetting_and_drying_alpha:
            min_alpha = self.options.wetting_and_drying_alpha_min
            max_alpha = self.options.wetting_and_drying_alpha_max

            # Take the dot product and threshold it
            alpha = dot(get_cell_widths_2d(self.mesh2d), abs(grad(self.fields.bathymetry_2d)))
            if max_alpha is not None:
                alpha = min_value(max_alpha, alpha)
            if min_alpha is not None:
                alpha = max_value(min_alpha, alpha)

            # Interpolate into P1 space
            self.options.wetting_and_drying_alpha = Function(self.function_spaces.P1_2d)
            self.options.wetting_and_drying_alpha.interpolate(alpha)

        # Print to screen and check validity
        alpha = self.options.wetting_and_drying_alpha
        if isinstance(alpha, Constant):
            msg = "Using constant wetting and drying parameter (value {:.2f})"
            assert alpha.values()[0] >= 0.0
            print_output(msg.format(alpha.values()[0]))
        elif isinstance(alpha, Function):
            msg = "Using spatially varying wetting and drying parameter (min {:.2f} max {:.2f})"
            with alpha.dat.vec_ro as v:
                alpha_min, alpha_max = v.min()[1], v.max()[1]
                assert alpha_min >= 0.0
                print_output(msg.format(alpha_min, alpha_max))
        else:
            msg = "Wetting and drying parameter of type '{:}' not supported"
            raise TypeError(msg.format(alpha.__class__.__name__))

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_function_spaces")
    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        on_the_sphere = self.mesh2d.geometric_dimension() == 3
        if on_the_sphere:
            assert self.options.element_family in ['rt-dg', 'bdm-dg'], \
                'Spherical mesh requires \'rt-dg\' or \'bdm-dg\' ' \
                'element family.'
        # ----- function spaces: elev in H, uv in U, mixed is W
        DG = 'DG' if self.mesh2d.ufl_cell().cellname() == 'triangle' else 'DQ'
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, DG, 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P1v_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1v_2d',
                                                        vector=True)
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, DG, 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = get_functionspace(self.mesh2d, DG, 1, name='P1DGv_2d',
                                                          vector=True)
        # 2D velocity space
        if self.options.element_family in ['rt-dg', 'bdm-dg']:
            family_prefix = self.options.element_family.split('-')[0].upper()
            family_suffix = {'triangle': 'F', 'quadrilateral': 'CF'}
            cell = self.mesh2d.ufl_cell().cellname()
            fam = family_prefix + family_suffix[cell]
            degree = self.options.polynomial_degree + 1
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, fam, degree, name='U_2d')
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, DG, self.options.polynomial_degree, name='H_2d')
        elif self.options.element_family == 'dg-cg':
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, DG, self.options.polynomial_degree, name='U_2d', vector=True)
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'CG', self.options.polynomial_degree+1, name='H_2d')
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, DG, self.options.polynomial_degree, name='U_2d', vector=True)
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, DG, self.options.polynomial_degree, name='H_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        if self.options.tracer_element_family == 'dg':
            self.function_spaces.Q_2d = get_functionspace(self.mesh2d, 'DG', 1, name='Q_2d')
        elif self.options.tracer_element_family == 'cg':
            self.function_spaces.Q_2d = get_functionspace(self.mesh2d, 'CG', 1, name='Q_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.tracer_element_family))

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.add_new_field")
    def add_new_field(self, function, label, name, filename, shortname=None, unit='-', preproc_func=None):
        """
        Add a field to :attr:`fields`.

        :arg function: representation of the field as a :class:`Function`
        :arg label: field label used internally by Thetis, e.g. 'tracer_2d'
        :arg name: human readable name for the tracer field, e.g. 'Tracer concentration'
        :arg filename: file name for outputs, e.g. 'Tracer2d'
        :kwarg shortname: short version of name, e.g. 'Tracer'
        :kwarg unit: units for field, e.g. '-'
        :kwarg preproc_func: optional pre-processor function which will be called before exporting
        """
        assert isinstance(function, Function)
        assert isinstance(label, str)
        assert isinstance(name, str)
        assert isinstance(filename, str)
        assert shortname is None or isinstance(shortname, str)
        assert isinstance(unit, str)
        assert preproc_func is None or callable(preproc_func)
        assert label not in field_metadata, f"Field '{label}' already exists."
        assert ' ' not in label, "Labels cannot contain spaces"
        assert ' ' not in filename, "Filenames cannot contain spaces"
        field_metadata[label] = {
            'name': name,
            'shortname': shortname or name,
            'unit': unit,
            'filename': filename,
        }
        self.fields[label] = function
        if preproc_func is not None:
            self._field_preproc_funcs[label] = preproc_func

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_fields")
    def create_fields(self):
        """
        Creates field Functions
        """
        if not hasattr(self.function_spaces, 'U_2d'):
            self.create_function_spaces()

        if self.options.log_output and not self.options.no_exports:
            mode = "a" if self.keep_log else "w"
            set_log_directory(self.options.output_directory, mode=mode)

        # Add general fields
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)
        self.set_wetting_and_drying_alpha()
        self.depth = DepthExpression(self.fields.bathymetry_2d,
                                     use_nonlinear_equations=self.options.use_nonlinear_equations,
                                     use_wetting_and_drying=self.options.use_wetting_and_drying,
                                     wetting_and_drying_alpha=self.options.wetting_and_drying_alpha)

        # Add fields for shallow water modelling
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        uv_2d, elev_2d = self.fields.solution_2d.subfunctions  # correct treatment of the split 2d functions
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d

        # Add tracer fields
        self.solve_tracer = len(self.options.tracer_fields.keys()) > 0
        for system, parent in self.options.tracer_fields.copy().items():
            labels = system.split(',')
            num_labels = len(labels)
            if parent is None:
                Q_2d = self.function_spaces.Q_2d
                fs = Q_2d if num_labels == 1 else MixedFunctionSpace([Q_2d]*len(labels))
                parent = Function(fs)
            if num_labels > 1:
                self.fields[system] = parent
            self.options.tracer_fields[system] = parent
            children = [parent] if num_labels == 1 else parent.subfunctions
            for label, function in zip(labels, children):
                tracer = self.options.tracer[label]
                function.rename(label)
                self.add_new_field(function,
                                   label,
                                   tracer.metadata['name'],
                                   tracer.metadata['filename'],
                                   shortname=tracer.metadata['shortname'],
                                   unit=tracer.metadata['unit'])

        # Add suspended sediment field
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.fields.sediment_2d = Function(self.function_spaces.Q_2d, name='sediment_2d')

        # Add fields for non-hydrostatic mode
        if self.options.nh_model_options.solve_nonhydrostatic_pressure:
            q_degree = self.options.polynomial_degree
            if self.options.nh_model_options.q_degree is not None:
                q_degree = self.options.nh_model_options.q_degree
            fs_q = get_functionspace(self.mesh2d, 'CG', q_degree)
            self.fields.q_2d = Function(fs_q, name='q_2d')  # 2D non-hydrostatic pressure at bottom
            self.fields.w_2d = Function(self.function_spaces.H_2d, name='w_2d')  # depth-averaged vertical velocity

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_equations")
    def create_equations(self):
        """
        Creates equation instances
        """
        if not hasattr(self.fields, 'uv_2d'):
            self.create_fields()
        self.equations = AttrDict()

        # tidal farms, if any
        if len(self.options.tidal_turbine_farms) + len(self.options.discrete_tidal_turbine_farms) > 0:
            self.tidal_farms = []
            p = self.function_spaces.U_2d.ufl_element().degree()
            quad_degree = 2*p + 1
            for subdomain, farm_options_list in self.options.tidal_turbine_farms.items():
                if not isinstance(farm_options_list, list):
                    error_msg = "Farm options must be entered as a list e.g. " \
                                "solver2d.FlowSolver2d(mesh2d, bathymetry_2d).options.discrete_tidal_turbine_farms[site_ID] = " \
                                "[farm_options]"
                    raise TypeError(error_msg)
                for farm_options in farm_options_list:
                    fdx = dx(subdomain, degree=quad_degree)
                    self.tidal_farms.append(TidalTurbineFarm(farm_options.turbine_density, fdx, farm_options))
            for subdomain, farm_options_list in self.options.discrete_tidal_turbine_farms.items():
                if not isinstance(farm_options_list, list):
                    error_msg = "Farm options must be entered as a list e.g. " \
                                "solver2d.FlowSolver2d(mesh2d, bathymetry_2d).options.discrete_tidal_turbine_farms[site_ID] = " \
                                "[farm_options]"
                    raise TypeError(error_msg)
                for farm_options in farm_options_list:
                    fdx = dx(subdomain, degree=farm_options.quadrature_degree)
                    self.tidal_farms.append(DiscreteTidalTurbineFarm(self.mesh2d, fdx, farm_options))
        else:
            self.tidal_farms = None
        # Shallow water equations for hydrodynamic modelling
        self.equations.sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.depth,
            self.options,
            tidal_farms=self.tidal_farms
        )
        self.equations.sw.bnd_functions = self.bnd_functions['shallow_water']
        uv_2d, elev_2d = self.fields.solution_2d.subfunctions

        # Passive tracer equations
        for system, parent in self.options.tracer_fields.items():
            self.equations[system] = tracer_eq_2d.TracerEquation2D(
                system,
                parent.function_space(),
                self.depth,
                self.options,
                uv_2d,
            )

        # Equation for suspended sediment transport
        sediment_options = self.options.sediment_model_options
        if sediment_options.solve_suspended_sediment or sediment_options.solve_exner:
            sediment_model_class = self.options.sediment_model_options.sediment_model_class
            self.sediment_model = sediment_model_class(
                self.options, self.mesh2d, uv_2d, elev_2d, self.depth)
        if sediment_options.solve_suspended_sediment:
            self.equations.sediment = sediment_eq_2d.SedimentEquation2D(
                self.function_spaces.Q_2d, self.depth, self.options, self.sediment_model,
                conservative=sediment_options.use_sediment_conservative_form)

        # Exner equation for bedload transport
        if sediment_options.solve_exner:
            if element_continuity(self.fields.bathymetry_2d.function_space().ufl_element()).horizontal in ['cg']:
                self.equations.exner = exner_eq.ExnerEquation(
                    self.fields.bathymetry_2d.function_space(), self.depth,
                    depth_integrated_sediment=sediment_options.use_sediment_conservative_form, sediment_model=self.sediment_model)
            else:
                raise NotImplementedError("Exner equation can currently only be implemented if the bathymetry is defined on a continuous space")

        # Free surface equation for non-hydrostatic pressure
        if self.options.nh_model_options.solve_nonhydrostatic_pressure:
            print_output('Using non-hydrostatic pressure')
            self.equations.fs = shallowwater_eq.FreeSurfaceEquation(
                TestFunction(self.function_spaces.H_2d), self.function_spaces.H_2d, self.function_spaces.U_2d,
                self.depth, self.options)
            self.equations.fs.bnd_functions = self.bnd_functions['shallow_water']

        # Setup slope limiters
        if self.solve_tracer or sediment_options.solve_suspended_sediment:
            if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
                self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.Q_2d)
            else:
                self.tracer_limiter = None

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.get_swe_timestepper")
    def get_swe_timestepper(self, integrator):
        """
        Gets shallow water timestepper object with appropriate parameters
        """
        fields = {
            'linear_drag_coefficient': self.options.linear_drag_coefficient,
            'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
            'manning_drag_coefficient': self.options.manning_drag_coefficient,
            'nikuradse_bed_roughness': self.options.nikuradse_bed_roughness,
            'viscosity_h': self.options.horizontal_viscosity,
            'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
            'coriolis': self.options.coriolis_frequency,
            'wind_stress': self.options.wind_stress,
            'atmospheric_pressure': self.options.atmospheric_pressure,
            'momentum_source': self.options.momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
        }
        bnd_conditions = self.bnd_functions['shallow_water']
        if self.options.swe_timestepper_type == 'PressureProjectionPicard':
            u_test = TestFunction(self.function_spaces.U_2d)
            self.equations.mom = shallowwater_eq.ShallowWaterMomentumEquation(
                u_test, self.function_spaces.U_2d, self.function_spaces.H_2d,
                self.depth,
                options=self.options,
                tidal_farms=self.tidal_farms
            )
            self.equations.mom.bnd_functions = bnd_conditions
            return integrator(self.equations.sw, self.equations.mom, self.fields.solution_2d,
                              fields, self.dt, self.options.swe_timestepper_options, bnd_conditions)
        else:
            return integrator(self.equations.sw, self.fields.solution_2d, fields, self.dt,
                              self.options.swe_timestepper_options, bnd_conditions)

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.get_tracer_timestepper")
    def get_tracer_timestepper(self, integrator, system):
        """
        Gets tracer timestepper object with appropriate parameters
        """
        uv, elev = self.fields.solution_2d.subfunctions
        fields = {
            'elev_2d': elev,
            'uv_2d': uv,
            'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
            'tracer_advective_velocity_factor': self.options.tracer_advective_velocity_factor,
        }
        for label in system.split(','):
            fields[f'diffusivity_h-{label}'] = self.options.tracer[label].diffusivity
            fields[f'source-{label}'] = self.options.tracer[label].source
        bcs = {}
        if system in self.bnd_functions:
            bcs = self.bnd_functions[system]
        elif system[:-3] in self.bnd_functions:
            # TODO: Is this safe for monolithic systems?
            bcs = self.bnd_functions[system[:-3]]
        # TODO: Different timestepper options for different systems
        return integrator(self.equations[system], self.fields[system], fields, self.dt,
                          self.options.tracer_timestepper_options, bcs)

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.get_sediment_timestepper")
    def get_sediment_timestepper(self, integrator):
        """
        Gets sediment timestepper object with appropriate parameters
        """
        uv, elev = self.fields.solution_2d.subfunctions
        fields = {
            'elev_2d': elev,
            'uv_2d': uv,
            'diffusivity_h-sediment_2d': self.options.sediment_model_options.horizontal_diffusivity,
            'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
            'tracer_advective_velocity_factor': self.sediment_model.get_advective_velocity_correction_factor(),
        }
        return integrator(self.equations.sediment, self.fields.sediment_2d, fields, self.dt,
                          self.options.sediment_model_options.sediment_timestepper_options,
                          self.bnd_functions['sediment'])

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.get_exner_timestepper")
    def get_exner_timestepper(self, integrator):
        """
        Gets exner timestepper object with appropriate parameters
        """
        uv, elev = self.fields.solution_2d.subfunctions
        if not self.options.sediment_model_options.solve_suspended_sediment:
            self.fields.sediment_2d = Function(elev.function_space()).interpolate(Constant(0.0))
        fields = {
            'elev_2d': elev,
            'sediment': self.fields.sediment_2d,
            'morfac': self.options.sediment_model_options.morphological_acceleration_factor,
            'porosity': self.options.sediment_model_options.porosity,
        }
        # only pass SWE bcs, used to determine closed boundaries in bedload term
        return integrator(self.equations.exner, self.fields.bathymetry_2d, fields, self.dt,
                          self.options.sediment_model_options.exner_timestepper_options,
                          self.bnd_functions['shallow_water'])

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.get_fs_timestepper")
    def get_fs_timestepper(self, integrator):
        """
        Gets free-surface correction timestepper object with appropriate parameters
        """
        fields_fs = {
            'uv': self.fields.uv_2d,
            'volume_source': self.options.volume_source_2d,
        }
        return integrator(self.equations.fs, self.fields.elev_2d, fields_fs, self.dt,
                          self.options.nh_model_options.free_surface_timestepper_options,
                          self.bnd_functions['shallow_water'])

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_timestepper")
    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        if not hasattr(self, 'equations'):
            self.create_equations()

        self.compute_mesh_stats()
        self.set_time_step()

        # ----- Time integrators
        steppers = {
            'SSPRK33': rungekutta.SSPRK33,
            'ForwardEuler': timeintegrator.ForwardEuler,
            'SteadyState': timeintegrator.SteadyState,
            'BackwardEuler': rungekutta.BackwardEulerUForm,
            'DIRK22': rungekutta.DIRK22UForm,
            'DIRK33': rungekutta.DIRK33UForm,
            'CrankNicolson': timeintegrator.CrankNicolson,
            'PressureProjectionPicard': timeintegrator.PressureProjectionPicard,
            'SSPIMEX': implicitexplicit.IMEXLPUM2,
        }
        if self.options.nh_model_options.solve_nonhydrostatic_pressure:
            self.poisson_solver = DepthIntegratedPoissonSolver(
                self.fields.q_2d, self.fields.uv_2d, self.fields.w_2d,
                self.fields.elev_2d, self.depth, self.dt, self.bnd_functions,
                solver_parameters=self.options.nh_model_options.solver_parameters
            )
            self.timestepper = coupled_timeintegrator_2d.NonHydrostaticTimeIntegrator2D(
                weakref.proxy(self), steppers[self.options.swe_timestepper_type],
                steppers[self.options.nh_model_options.free_surface_timestepper_type]
            )
        elif self.solve_tracer:
            self.timestepper = coupled_timeintegrator_2d.GeneralCoupledTimeIntegrator2D(
                weakref.proxy(self), {
                    'shallow_water': steppers[self.options.swe_timestepper_type],
                    'tracer': steppers[self.options.tracer_timestepper_type],
                },
            )
        elif self.options.sediment_model_options.solve_suspended_sediment or self.options.sediment_model_options.solve_exner:
            self.timestepper = coupled_timeintegrator_2d.GeneralCoupledTimeIntegrator2D(
                weakref.proxy(self), {
                    'shallow_water': steppers[self.options.swe_timestepper_type],
                    'sediment': steppers[self.options.sediment_model_options.sediment_timestepper_type],
                    'exner': steppers[self.options.sediment_model_options.exner_timestepper_type],
                },
            )
        else:
            self.timestepper = self.get_swe_timestepper(steppers[self.options.swe_timestepper_type])
        print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_exporters")
    def create_exporters(self):
        """
        Creates file exporters
        """
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        self.exporters = OrderedDict()
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.output_directory,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0,
                                       preproc_funcs=self._field_preproc_funcs)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.output_directory, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0,
                                       preproc_funcs=self._field_preproc_funcs)
            self.exporters['hdf5'] = e

    def initialize(self):
        """
        Creates function spaces, equations, time stepper and exporters
        """
        if not hasattr(self.function_spaces, 'U_2d'):
            self.create_function_spaces()
        if not hasattr(self, 'equations'):
            self.create_equations()
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        if not hasattr(self, 'exporters'):
            self.create_exporters()
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.assign_initial_conditions")
    def assign_initial_conditions(self, elev=None, uv=None, sediment=None, **tracers):
        r"""
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        :kwarg sediment: Initial condition for sediment concantration
        :type sediment: scalar valued :class:`Function`, :class:`Constant`, or an expression
        :kwarg tracers: Initial conditions for tracer fields
        :type tracers: scalar valued :class:`Function`\s, :class:`Constant`\s, or an expressions
        """
        if not self._initialized:
            self.initialize()
        uv_2d, elev_2d = self.fields.solution_2d.subfunctions
        if elev is not None:
            elev_2d.project(elev)
        if uv is not None:
            uv_2d.project(uv)
        for l, func in tracers.items():
            label = l if len(l) > 3 and l[-3:] == '_2d' else l + '_2d'
            assert label in self.options.tracer, f"Unknown tracer label {label}"
            self.fields[label].project(func)

        sediment_options = self.options.sediment_model_options
        if self.sediment_model is not None:
            # update sediment model based on initial conditions for uv and elev
            self.sediment_model.update()
        if sediment_options.solve_suspended_sediment:
            if sediment is not None:
                self.fields.sediment_2d.project(sediment)
            else:
                sediment = self.sediment_model.get_equilibrium_tracer()
                if sediment_options.use_sediment_conservative_form:
                    sediment = sediment * self.depth.get_total_depth(elev_2d)
                self.fields.sediment_2d.project(sediment)

        self.timestepper.initialize(self.fields.solution_2d)

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.add_callback")
    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg string eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.export")
    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export')
        for e in self.exporters.values():
            e.export()

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.load_state")
    def load_state(self, i_stored, outputdir=None, t=None, iteration=None,
                   i_export=None, legacy_mode=False):
        """
        Loads simulation state from hdf5 outputs.

        Replaces :meth:`.assign_initial_conditions` in model initialization.

        For example, continue simulation from export 40,

        .. code-block:: python

            solver_obj.load_state(40)

        Continue simulation from another output directory,

        .. code-block:: python

            solver_obj.load_state(40, outputdir='outputs_spinup')

        Continue simulation from another output directory and reset the
        counters,

        .. code-block:: python

            solver_obj.load_state(40, outputdir='outputs_spinup',
                iteration=0, t=0, i_export=0)

        Model state can be loaded if all prognostic state variables have been
        stored in hdf5 format. For the hydrodynamics the required state
        variables are: `elev_2d`, and `uv_2d`.

        Simulation time and iteration can be recovered automatically provided
        that the model setup is the same (e.g. time step and export_time).

        :arg int i_stored: export index to load
        :kwarg string outputdir: (optional) directory where files are read from.
            By default ``options.output_directory``.
        :kwarg float t: simulation time. Overrides the time stamp inferred from
            model options.
        :kwarg int iteration: Overrides the iteration count inferred from model
            options.
        :kwarg int i_export: Set initial export index for the present run. By
            default, `i_stored` is used.
        :kwarg bool legacy_mode: Load legacy `DumbCheckpoint` files.
        """
        if not self._initialized:
            self.initialize()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   legacy_mode=legacy_mode,
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_stored, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_stored, self.fields.elev_2d)
        self.assign_initial_conditions()

        # time stepper bookkeeping for export time step
        if i_export is None:
            i_export = i_stored
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.simulation_export_time
        if iteration is None:
            iteration = int(numpy.ceil(self.next_export_t/self.dt))
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

    def print_state(self, cputime, print_header=False):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time in seconds
        :kwarg print_header: Whether to print column header first
        """
        # list of entries to print: (header, value, format)
        entries = [
            ('exp', self.i_export, '5d'),
            ('iter', self.iteration, '5d'),
        ]
        # generate simulation time string
        if self.options.simulation_initial_date is not None:
            now = self.options.simulation_initial_date + datetime.timedelta(seconds=self.simulation_time)
            date_str = f'{now:%Y-%m-%d}'.rjust(11)
            time_str = f'{now:%H:%M:%S}'.rjust(9)
            entries += [
                ('date', date_str, '11s'),
                ('time', time_str, '9s'),
            ]
        else:
            time_str = f'{self.simulation_time:.2f}'.rjust(15)
            entries += [
                ('time', time_str, '15s'),
            ]
        if self.options.tracer_only:
            for label in self.options.tracer:
                norm_q = norm(self.fields[label])
                e = (label, norm_q, '10.4f')
                entries.append(e)
        else:
            norm_h = norm(self.fields.solution_2d.subfunctions[1])
            norm_u = norm(self.fields.solution_2d.subfunctions[0])
            entries += [
                ('eta norm', norm_h, '14.4f'),
                ('u norm', norm_u, '14.4f'),
            ]
        entries.append(('Tcpu', cputime, '6.2f'))

        if print_header:
            # generate header
            header = ' '.join([e[0].rjust(len(f'{e[1]:{e[2]}}')) for e in entries])
            print_output(header)

        # generate line
        line = ' '.join([f'{e[1]:{e[2]}}' for e in entries])
        print_output(line)
        sys.stdout.flush()

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.iterate")
    def iterate(self, update_forcings=None, export_func=None):
        """
        Runs the simulation

        Wrapper for function for `create_iterator` generator and automatically
        iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        for _ in self.create_iterator(update_forcings=update_forcings,
                                      export_func=export_func):
            pass

    @PETSc.Log.EventDecorator("thetis.FlowSolver2d.create_iterator")
    def create_iterator(self, update_forcings=None, export_func=None):
        """
        Creates a generator to iterate through the simulation and return access
        to time advancing function when time control is handled externally.

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        For example:

        .. code-block:: python

            for t in solver_obj.create_iterator():
                # user code

        or, to get per time-step control:

        .. code-block:: python

            thetis_timestepper = solver_obj.create_iterator()

            while t_Thetis<t_end - timestep and .... :
                t_Thetis = next(thetis_timestepper)

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """

        if not self._initialized:
            self.initialize()

        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.perf_counter()
        next_export_t = self.simulation_time + self.options.simulation_export_time

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        if self.options.check_tracer_conservation:
            for label, tracer in self.options.tracer.items():
                if tracer.use_conservative_form:
                    c = callback.ConservativeTracerMassConservation2DCallback(label,
                                                                              self,
                                                                              export_to_hdf5=dump_hdf5,
                                                                              append_to_log=True)
                else:
                    c = callback.TracerMassConservation2DCallback(label,
                                                                  self,
                                                                  export_to_hdf5=dump_hdf5,
                                                                  append_to_log=True)
                self.add_callback(c, eval_interval='export')
        if self.options.sediment_model_options.check_sediment_conservation:
            if self.options.sediment_model_options.use_sediment_conservative_form:
                c = callback.ConservativeTracerMassConservation2DCallback('sediment_2d',
                                                                          self,
                                                                          export_to_hdf5=dump_hdf5,
                                                                          append_to_log=True)
            else:
                c = callback.TracerMassConservation2DCallback('sediment_2d',
                                                              self,
                                                              export_to_hdf5=dump_hdf5,
                                                              append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self.options.check_tracer_overshoot:
            for label in self.options.tracer:
                c = callback.TracerOvershootCallBack(label,
                                                     self,
                                                     export_to_hdf5=dump_hdf5,
                                                     append_to_log=True)
                self.add_callback(c, eval_interval='export')
        if self.options.sediment_model_options.check_sediment_overshoot:
            c = callback.TracerOvershootCallBack('sediment_2d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        init_date = self.options.simulation_initial_date
        end_date = self.options.simulation_end_date
        if (init_date is not None and end_date is not None):
            now = init_date + datetime.timedelta(initial_simulation_time)
            assert end_date > now, f'Simulation end date must be greater than initial time {now}'
            print_output(
                f'Running simulation\n'
                f' from {now:%Y-%m-%d %H:%M:%S %Z}\n'
                f' to   {end_date:%Y-%m-%d %H:%M:%S %Z}'
            )
            end_time = (end_date - now).total_seconds() + initial_simulation_time
            if self.options.simulation_end_time is not None:
                warning('Both simulation_end_date and simulation_end_time have been set, ignoring simulation_end_time.')
            self.options.simulation_end_time = end_time
        assert self.options.simulation_end_time is not None, 'simulation_end_time must be set'

        # initial export
        self.print_state(0.0, print_header=True)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters and isinstance(self.fields.bathymetry_2d, Function):
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:
            self.timestepper.advance(self.simulation_time, update_forcings)

            # returns internal simulation time
            yield self.simulation_time

            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                self.i_export += 1
                next_export_t += self.options.simulation_export_time

                cputime = time_mod.perf_counter() - cputimestamp
                cputimestamp = time_mod.perf_counter()
                self.print_state(cputime)

                self.export()
                if export_func is not None:
                    export_func()
