# TODO: doc
from __future__ import absolute_import
from .shallowwater_eq import *
from .utility import *


g_grav = physical_constants['g_grav']


class AdjointBCMixin:
    # TODO: doc

    def get_bnd_functions(self, zeta_in, z_in, bnd_id, bnd_conditions):
        homogeneous = False
        funcs = bnd_conditions.get(bnd_id)
        if funcs is not None and 'elev' not in funcs and 'un' not in funcs:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))

        if homogeneous:
            if 'elev' in funcs and 'un' in funcs:
                zeta_ext = Constant(0.0)
                z_ext = Constant(0.0)*self.normal
            elif 'elev' in funcs:
                zeta_ext = Constant(0.0)
                z_ext = z_in  # assume symmetry
            elif 'un' in funcs:
                zeta_ext = zeta_in  # assume symmetry
                z_ext = Constant(0.0)*self.normal
            else:
                zeta_ext = zeta_in  # assume symmetry
                z_ext = z_in  # assume symmetry
        else:
            if 'elev' in funcs and 'un' in funcs:
                zeta_ext = zeta_in  # assume symmetry
                z_ext = z_in  # assume symmetry
            elif 'elev' not in funcs:
                zeta_ext = zeta_in  # assume symmetry
                z_ext = Constant(0.0)*self.normal
            elif 'un' not in funcs:
                zeta_ext = Constant(0.0)
                z_ext = z_in  # assume symmetry
            else:
                zeta_ext = Constant(0.0)
                z_ext = Constant(0.0)*self.normal

        return zeta_ext, z_ext


class AdjointExternalPressureGradientTerm(AdjointBCMixin, ShallowWaterContinuityTerm):
    raise NotImplementedError  # TODO


class AdjointHUDivMomentumTerm(AdjointBCMixin, ShallowWaterMomentumTerm):
    raise NotImplementedError  # TODO


class AdjointHUDivContinuityTerm(AdjointBCMixin, HUDivTerm):
    raise NotImplementedError  # TODO


class AdjointHorizontalAdvectionTerm(AdjointBCMixin, HorizontalAdvectionTerm):
    raise NotImplementedError  # TODO


class AdjointHorizontalViscosityTerm(AdjointBCMixin, HorizontalViscosityTerm):
    """
    Viscosity is identical to that in the forward model.
    """
    pass


class AdjointCoriolisTerm(CoriolisTerm):
    """
    The Coriolis term is identical to that in the forward model.
    """
    pass


class AdjointLinearDragTerm(LinearDragTerm):
    """
    The linear drag term is identical to that in the forward model.
    """
    pass


class AdjointWindStressTerm(ShallowWaterContinuityTerm):
    raise NotImplementedError  # TODO


class AdjointQuadraticDragMomentumTerm(ShallowWaterMomentumTerm):
    raise NotImplementedError  # TODO


class AdjointQuadraticDragContinuityTerm(ShallowWaterContinuityTerm):
    raise NotImplementedError  # TODO


class AdjointBottomDrag3DTerm(ShallowWaterContinuityTerm):
    raise NotImplementedError  # TODO


class AdjointTurbineDragMomentumTerm(ShallowWaterMomentumTerm):
    raise NotImplementedError  # TODO


class AdjointTurbineDragContinuityTerm(ShallowWaterContinuityTerm):
    raise NotImplementedError  # TODO


class AdjointMomentumSourceTerm(MomentumSourceTerm):
    r"""
    Term on the right hand side of the adjoint momentum equation corresponding to the derivative of
    the quantity of interest :math:`J` with respect to velocity, :math:`\partial J/\partial u`.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        dJdu = fields_old.get('dJdu')

        f = 0
        if dJdu is not None:
            f += inner(dJdu, self.u_test)*self.dx
        return f


class AdjointContinuitySourceTerm(ContinuitySourceTerm):
    r"""
    Term on the right hand side of the adjoint continuity equation corresponding to the derivative of
    the quantity of interest :math:`J` with respect to elevation, :math:`\partial J/\partial\eta`.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        dJdeta = fields_old.get('dJdeta')

        f = 0
        if dJdeta is not None:
            f += inner(dJdeta, self.eta_test)*self.dx
        return f


class AdjointBathymetryDisplacementMassTerm(BathymetryDisplacementMassTerm):
    raise NotImplementedError  # TODO


class BaseAdjointShallowWaterEquation(BaseShallowWaterEquation):
    """
    Abstract base class for continuous adjoint shallow water formulations.
    """
    def add_momentum_terms(self, *args):
        self.add_term(AdjointHUDivMomentumTerm(*args), 'implicit')
        self.add_term(AdjointHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(AdjointHorizontalViscosityTerm(*args), 'explicit')
        self.add_term(AdjointCoriolisTerm(*args), 'explicit')
        self.add_term(AdjointLinearDragTerm(*args), 'explicit')
        self.add_term(AdjointQuadraticDragMomentumTerm(*args), 'explicit')
        self.add_term(AdjointTurbineDragMomentumTerm(*args), 'implicit')
        self.add_term(AdjointBottomDrag3DTerm(*args), 'source')
        self.add_term(AdjointMomentumSourceTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(AdjointHUDivContinuityTerm(*args), 'implicit')
        self.add_term(AdjointExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(AdjointQuadraticDragContinuityTerm(*args), 'explicit')
        self.add_term(AdjointTurbineDragContinuityTerm(*args), 'implicit')
        self.add_term(AdjointContinuitySourceTerm(*args), 'source')


class AdjointShallowWaterEquations(BaseAdjointShallowWaterEquation):
    """
    Continuous adjoint formulation of 2D depth-averaged shallow water equations in non-conservative
    form.
    """
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterEquations, self).__init__(function_space, depth, options)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space, depth, options)

        self.add_continuity_terms(eta_test, eta_space, u_space, depth, options)
        self.bathymetry_displacement_mass_term = AdjointBathymetryDisplacementMassTerm(
            eta_test, eta_space, u_space, depth, options)
