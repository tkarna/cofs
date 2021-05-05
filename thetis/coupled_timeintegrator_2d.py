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
            for label in self.options.tracer_data_2d:
                self.timesteppers[label] = self.solver.get_tracer_timestepper(self.tracer_integrator, label)
            if 'tracer_2d' in self.options.tracer_data_2d:
                self.timesteppers.tracer = self.timesteppers.tracer_2d  # Backwards compatibility
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
            for label in self.options.tracer_data_2d:
                self.timesteppers[label].initialize(self.fields[label])
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.initialize(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.initialize(self.fields.bathymetry_2d)

        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self.options.tracer_only:
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
        if self.options.solve_tracer:
            for label in self.options.tracer_data_2d:
                self.timesteppers[label].advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                for label in self.options.tracer_data_2d:
                    self.solver.tracer_limiter.apply(self.fields[label])
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
