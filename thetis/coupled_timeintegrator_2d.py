"""
Time integrators for solving coupled shallow water equations with one tracer or sediment.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from abc import ABCMeta


class CoupledTimeIntegrator2D(timeintegrator.TimeIntegratorBase):
    """
    Base class of time integrator for coupled shallow water and tracer/sediment equations and exner equation
    """
    __metaclass__ = ABCMeta

    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

    def tracer_integrator(self):
        """time integrator for the tracer equation"""
        pass

    def exner_integrator(self):
        """time integrator for the exner equation"""
        pass

    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.fields = solver.fields
        self.timesteppers = AttrDict()
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        print_output('  Shallow Water time integrator: {:}'.format(self.swe_integrator.__name__))
        if self.options.solve_tracer:
            print_output('  Tracer time integrator: {:}'.format(self.tracer_integrator.__name__))
        if self.options.sediment_model_options.solve_suspended_sediment:
            print_output('  Sediment time integrator: {:}'.format(self.sediment_integrator.__name__))
        if self.options.sediment_model_options.solve_exner:
            print_output('  Exner time integrator: {:}'.format(self.exner_integrator.__name__))
        self._initialized = False

        self._create_integrators()

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        self.timesteppers.swe2d = self.solver.get_swe_timestepper(self.swe_integrator)
        if self.solver.options.solve_tracer:
            self.timesteppers.tracer = self.solver.get_tracer_timestepper(self.tracer_integrator)
        if self.solver.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment = self.solver.get_sediment_timestepper(self.sediment_integrator)
        if self.solver.options.sediment_model_options.solve_exner:
            self.timesteppers.exner = self.solver.get_exner_timestepper(self.exner_integrator)

    def set_dt(self, dt):
        """
        Set time step for the coupled time integrator

        :arg float dt: Time step.
        """
        for stepper in sorted(self.timesteppers):
            self.timesteppers[stepper].set_dt(dt)

    def initialize(self, solution2d):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """

        # solution2d is only provided to make a timeintegrator of just the swe
        # compatible with the 2d coupled timeintegrator
        assert solution2d == self.fields.solution_2d

        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.options.solve_tracer:
            self.timesteppers.tracer.initialize(self.fields.tracer_2d)
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.initialize(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.initialize(self.fields.bathymetry_2d)

        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self.options.tracer_only:
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
        if self.options.solve_tracer:
            self.timesteppers.tracer.advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.tracer_2d)
        if self.solver.sediment_model is not None:
            self.solver.sediment_model.update()
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.advance(t, update_forcings=update_forcings)


class CoupledMatchingTimeIntegrator2D(CoupledTimeIntegrator2D):
    def __init__(self, solver, integrator):
        self.swe_integrator = integrator
        if solver.options.solve_tracer:
            self.tracer_integrator = integrator
        if solver.options.sediment_model_options.solve_suspended_sediment:
            self.sediment_integrator = integrator
        if solver.options.sediment_model_options.solve_exner:
            self.exner_integrator = integrator
        super(CoupledMatchingTimeIntegrator2D, self).__init__(solver)


class CoupledCrankEuler2D(CoupledTimeIntegrator2D):
    swe_integrator = timeintegrator.CrankNicolson
    tracer_integrator = timeintegrator.ForwardEuler
    sediment_integrator = timeintegrator.ForwardEuler
    exner_integrator = timeintegrator.CrankNicolson


class NonHydrostaticTimeIntegrator2D(CoupledTimeIntegrator2D):
    """
    2D non-hydrostatic time integrator based on Shallow Water time integrator

    This time integration method uses SWE time integrator to advance
    the hydrostatic equations, depth-integrated Poisson solver to be
    solved for NH pressure, and a free surface integrator to advance
    the free surface correction. Advancing in serial or in a whole
    time stepping depends on the selection of time integrators.
    """
    def __init__(self, solver, swe_integrator, fs_integrator):
        self.swe_integrator = swe_integrator
        super(NonHydrostaticTimeIntegrator2D, self).__init__(solver)
        self.poisson_solver = solver.poisson_solver
        print_output('  Non-hydrostatic pressure solver: {:}'.format(self.poisson_solver.__class__.__name__))
        print_output('  Free Surface time integrator: {:}'.format(fs_integrator.__name__))
        self.nh_options = solver.options.nh_model_options
        if self.nh_options.update_free_surface:
            self.timesteppers.fs2d = self.solver.get_fs_timestepper(fs_integrator)
            self.elev_old = Function(self.fields.elev_2d)
        self.serial_advancing = not hasattr(self.timesteppers.swe2d, 'n_stages') \
            or self.options.timestepper_type == 'SSPIMEX'
        self.multi_stages_fs = hasattr(self.timesteppers.fs2d, 'n_stages') \
            and self.nh_options.free_surface_timestepper_type != 'BackwardEuler'
        if self.multi_stages_fs:
            msg = 'The multi-stage type of Shallow Water and ' \
                  'Free Surface time integrators should be the same.'
            assert self.options.timestepper_type == self.nh_options.free_surface_timestepper_type, msg

    def initialize(self, solution2d):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        assert solution2d == self.fields.solution_2d

        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.nh_options.update_free_surface:
            self.timesteppers.fs2d.initialize(self.fields.elev_2d)
            self.elev_old.assign(self.fields.elev_2d)

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if self.serial_advancing:
            # --- advance in serial ---
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
            # solve non-hydrostatic pressure q and update velocities
            self.poisson_solver.solve()
            # update free surface elevation
            if self.nh_options.update_free_surface:
                self.fields.elev_2d.assign(self.elev_old)
                self.timesteppers.fs2d.advance(t, update_forcings=update_forcings)
                self.elev_old.assign(self.fields.elev_2d)
            # update old solution
            if self.options.timestepper_type == 'SSPIMEX':
                self.timesteppers.swe2d.erk.solution_old.assign(self.fields.solution_2d)
                self.timesteppers.swe2d.dirk.solution_old.assign(self.fields.solution_2d)
        else:
            # --- advance in a whole stepping ---
            for i in range(self.timesteppers.swe2d.n_stages):
                last_stage = i == self.timesteppers.swe2d.n_stages - 1
                self.timesteppers.swe2d.solve_stage(i, t, update_forcings)
                # solve non-hydrostatic pressure q and update velocities
                self.poisson_solver.solve(solve_w=last_stage)
                # update free surface elevation
                if self.nh_options.update_free_surface:
                    if self.multi_stages_fs:
                        self.fields.elev_2d.assign(self.elev_old)
                        self.timesteppers.fs2d.solve_stage(i, t, update_forcings)
                        self.elev_old.assign(self.fields.elev_2d)
                    elif last_stage:
                        self.fields.elev_2d.assign(self.elev_old)
                        self.timesteppers.fs2d.advance(t, update_forcings=update_forcings)
                        self.elev_old.assign(self.fields.elev_2d)
