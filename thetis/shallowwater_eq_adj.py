# TODO: doc
from __future__ import absolute_import
from .shallowwater_eq import *
from .utility import *


g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


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
    r"""
    Term resulting from differentiating the external pressure gradient term with respect
    to elevation.

    Where the original term appeared in the momentum equation of the forward model, this
    term appears in the continuity equation of the adjoint model.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))

        z_by_parts = self.u_continuity in ['dg', 'hdiv']

        n = self.normal
        if z_by_parts:
            f = -g_grav*inner(grad(self.eta_test), z)*self.dx
            if self.eta_is_dg:
                h_av = avg(total_h)
                z_star = avg(z) + sqrt(h_av/g_grav)*jump(zeta, n)
                f += g_grav*inner(jump(self.eta_test, n), z_star)*dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                zeta_ext, z_ext = self.get_bnd_functions(zeta, z, bnd_marker, bnd_conditions)

                # Compute linear Riemann solution with zeta, zeta_ext, z, z_ext
                zeta_jump = zeta - zeta_ext
                zn_rie = 0.5*inner(z + z_ext, n) + sqrt(total_h/g_grav)*zeta_jump
                f += g_grav*self.eta_test*zn_rie*ds_bnd
        else:
            f = g_grav*self.eta_test*div(z)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None and 'elev' not in funcs:
                    f += -g_grav*dot(z, self.normal)*self.eta_test*ds_bnd

        return -f


class AdjointHUDivMomentumTerm(AdjointBCMixin, ShallowWaterMomentumTerm):
    r"""
    Term resulting from differentiating the :math:`\nabla\cdot(H\mathbf u)` term with respect to
    velocity. Note that, where the original term appeared in the continuity equation of the forward
    model, this term appears in the momentum equation of the adjoint model.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))

        zeta_by_parts = self.eta_is_dg

        f = 0
        n = self.normal
        if zeta_by_parts:
            f += -zeta*div(total_h*self.u_test)*self.dx
            h_av = avg(total_h)
            zeta_star = avg(zeta) + sqrt(g_grav/h_av)*jump(z, n)
            f += h_av*zeta_star*jump(self.u_test, n)*dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    zeta_ext, z_ext = self.get_bnd_functions(zeta, z, bnd_marker, bnd_conditions)

                    # Compute linear riemann solution with zeta, zeta_ext, z, z_ext
                    zn_jump = inner(z - z_ext, n)
                    zeta_rie = 0.5*(zeta + zeta_ext) + sqrt(g_grav/total_h)*zn_jump
                    f += total_h*zeta_rie*dot(self.u_test, n)*ds_bnd
        else:
            f += total_h*inner(grad(zeta), self.u_test)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    zeta_ext, z_ext = self.get_bnd_functions(zeta, z, bnd_marker, bnd_conditions)

                    # Compute linear riemann solution with zeta, zeta_ext, z, z_ext
                    zn_jump = inner(z - z_ext, n)
                    zeta_rie = 0.5*(zeta + zeta_ext) + sqrt(g_grav/total_h)*zn_jump
                    f += total_h*(zeta_rie-zeta)*dot(self.u_test, n)*ds_bnd
        return -f


class AdjointHUDivContinuityTerm(AdjointBCMixin, HUDivTerm):
    r"""
    Term resulting from differentiating the :math:`\nabla\cdot(H\mathbf u)` term with respect to
    elevation.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0

        zeta_by_parts = self.eta_is_dg

        # Account for wetting and drying
        test = self.eta_test
        if self.options.get('use_wetting_and_drying'):
            test = test*self.depth.heaviside_approx(self.options.get('elev_2d'))

        f = 0
        uv = fields.get('uv_2d')
        n = self.normal
        if zeta_by_parts:
            f += -zeta*div(test*uv)*self.dx
            f += inner(jump(self.eta_test, n), avg(zeta)*avg(uv))*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    zeta_ext, z_ext = self.get_bnd_functions(zeta, z, bnd_marker, bnd_conditions)
                    zeta_ext_old, z_ext_old = self.get_bnd_functions(zeta_old, z_old, bnd_marker, bnd_conditions)

                    # Compute linear riemann solution with zeta, zeta_ext, z, z_ext
                    zn_jump = inner(z_old - z_ext_old, n)
                    zeta_rie = 0.5*(zeta_old + zeta_ext_old) + sqrt(h_av/g_grav)*zn_jump
                    f += zeta_rie*dot(uv, self.normal)*self.eta_test*ds_bnd
        else:
            f += inner(grad(zeta), test*uv)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None and 'elev' not in funcs:
                    f += -zeta*dot(uv, self.normal)*test*ds_bnd


class AdjointHorizontalAdvectionTerm(AdjointBCMixin, HorizontalAdvectionTerm):
    """
    Term resulting from differentiating the nonlinear advection term by velocity.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0

        horiz_advection_by_parts = True

        uv = fields.get('uv_2d')
        un = 0.5*(abs(dot(uv, self.normal)) - dot(uv, self.normal))  # u.n if u.n < 0 else 0
        downwind = lambda x: conditional(un < 0, dot(x, self.normal), 0)

        f = 0
        f += -inner(dot(self.u_test, nabla_grad(uv)), z)*self.dx
        f += -inner(dot(uv, nabla_grad(self.u_test)), z)*self.dx
        if horiz_advection_by_parts:
            f += inner(jump(self.u_test), 2*avg(un*z))*self.dS
            f += inner(2*avg(downwind(self.u_test)*z), jump(uv))*self.dS
            # TODO: Boundary contributions?
            # TODO: Stabilisation?

        return -f


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
    """
    Term resulting from differentiating the wind stress term by elevation.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        wind_stress = fields_old.get('wind_stress')
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))
        f = 0
        if wind_stress is not None:
            f += -self.eta_test*dot(wind_stress, z)/total_h**2/rho_0*self.dx
        return f


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
