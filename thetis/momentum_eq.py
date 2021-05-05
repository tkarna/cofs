r"""
3D momentum equation for hydrostatic Boussinesq flow.

The three dimensional momentum equation reads

.. math::
    \frac{\partial \textbf{u}}{\partial t}
        + \nabla_h \cdot (\textbf{u} \textbf{u})
        + \frac{\partial \left(w\textbf{u} \right)}{\partial z}
        + f\textbf{e}_z\wedge\textbf{u} + g\nabla_h \eta + g\nabla_h r
        = \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)
        + \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right)
    :label: mom_eq

where :math:`\textbf{u}` and :math:`w` denote horizontal and vertical velocity,
:math:`\nabla_h` is the horizontal gradient,
:math:`\wedge` denotes the cross product,
:math:`g` is the gravitational acceleration, :math:`f` is the Coriolis
frequency, :math:`\textbf{e}_z` is a vertical unit vector, and
:math:`\nu_h, \nu` stand for horizontal and vertical viscosity.
Water density is given by :math:`\rho = \rho'(T, S, p) + \rho_0`,
where :math:`\rho_0` is a constant reference density.
Above :math:`r` denotes the baroclinic head

.. math::
    r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.
    :label: baroc_head

The internal pressure gradient is computed as a separate diagnostic field:

.. math::
    \mathbf{F}_{pg} = g\nabla_h r.
    :label: int_pg_eq

In the case of purely barotropic problems the :math:`r` and
:math:`\mathbf{F}_{pg}` fields are omitted.

When using mode splitting we split the velocity field into a depth average and
a deviation, :math:`\textbf{u} = \bar{\textbf{u}} + \textbf{u}'`.
Following Higdon and de Szoeke (1997) we write an equation for the deviation
:math:`\textbf{u}'`:

.. math::
    \frac{\partial \textbf{u}'}{\partial t} =
        + \nabla_h \cdot (\textbf{u} \textbf{u})
        + \frac{\partial \left(w\textbf{u} \right)}{\partial z}
        + f\textbf{e}_z\wedge\textbf{u}' + g\nabla_h r
        = \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)
        + \frac{\partial }{\partial z}\left( \nu  \frac{\partial \textbf{u}}{\partial z}\right)
    :label: mom_eq_split

In :eq:`mom_eq_split` the external pressure gradient :math:`g\nabla_h \eta` vanishes and the
Coriolis term only contains the deviation :math:`\textbf{u}'`.
Advection and diffusion terms are not changed.

Higdon and de Szoeke (1997). Barotropic-Baroclinic Time Splitting for Ocean
Circulation Modeling. Journal of Computational Physics, 135(1):30-53.
http://dx.doi.org/10.1006/jcph.1997.5733
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'MomentumEquation',
    'MomentumTerm',
    'HorizontalAdvectionTerm',
    'VerticalAdvectionTerm',
    'HorizontalViscosityTerm',
    'VerticalViscosityTerm',
    'PressureGradientTerm',
    'CoriolisTerm',
    'BottomFrictionTerm',
    'LinearDragTerm',
    'SourceTerm',
    'InternalPressureGradientCalculator',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class MomentumTerm(Term):
    """
    Generic term for momentum equation that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_nonlinear_equations=True, use_lax_friedrichs=True,
                 use_bottom_friction=False, sipg_factor=Constant(1.0),
                 sipg_factor_vertical=Constant(1.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_nonlinear_equations: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        :kwarg sipg_factor: :class: `Constant` or :class: `Function` horizontal SIPG penalty scaling factor
        :kwarg sipg_factor_vertical: :class: `Constant` or :class: `Function` vertical SIPG penalty scaling factor
        """
        super(MomentumTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_continuity = continuity.horizontal
        self.vertical_continuity = continuity.vertical
        self.use_nonlinear_equations = use_nonlinear_equations
        self.use_lax_friedrichs = use_lax_friedrichs
        self.use_bottom_friction = use_bottom_friction
        self.sipg_factor = sipg_factor
        self.sipg_factor_vertical = sipg_factor_vertical

        # define measures with a reasonable quadrature degree
        p, q = self.function_space.ufl_element().degree()
        self.quad_degree = (2*p + 1, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

        # TODO add generic get_bnd_functions?


class PressureGradientTerm(MomentumTerm):
    r"""
    Internal pressure gradient term, :math:`g\nabla_h r`

    where :math:`r` is the baroclinic head :eq:`baroc_head`. Let :math:`s`
    denote :math:`r/H`. We can then write

    .. math::
        F_{IPG} = g\nabla_h((s -\bar{s}) H)
            + g\nabla_h\left(\frac{1}{H}\right) H^2\bar{s}
            + g s_{bot}\nabla_h h

    where :math:`\bar{s},s_{bot}` are the depth average and bottom value of
    :math:`s`.

    If :math:`s` belongs to a discontinuous function space, the first term is
    integrated by parts. Its weak form reads

    .. math::
        \int_\Omega g\nabla_h((s -\bar{s}) H) \cdot \boldsymbol{\psi} dx
            = - \int_\Omega g (s -\bar{s}) H \nabla_h \cdot \boldsymbol{\psi} dx
            + \int_{\mathcal{I}_h \cup \mathcal{I}_v} g (s -\bar{s}) H \boldsymbol{\psi}  \cdot \textbf{n}_h dx
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        int_pg = fields.get('int_pg')
        if int_pg is None:
            return 0

        f = (int_pg[0]*self.test[0] + int_pg[1]*self.test[1])*self.dx

        return -f


class HorizontalAdvectionTerm(MomentumTerm):
    r"""
    Horizontal advection term, :math:`\nabla_h \cdot (\textbf{u} \textbf{u})`

    The weak form reads

    .. math::
        \int_\Omega \nabla_h \cdot (\textbf{u} \textbf{u}) \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \nabla_h \boldsymbol{\psi} : (\textbf{u} \textbf{u}) dx
        + \int_{\mathcal{I}_h \cup \mathcal{I}_v} \textbf{u}^{\text{up}} \cdot
        \text{jump}(\boldsymbol{\psi} \textbf{n}_h) \cdot \text{avg}(\textbf{u}) dS

    where the right hand side has been integrated by parts; :math:`:` stand for
    the Frobenius inner product, :math:`\textbf{n}_h` is the horizontal
    projection of the normal vector, :math:`\textbf{u}^{\text{up}}` is the
    upwind value, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if not self.use_nonlinear_equations:
            return 0
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
            uv_old = solution_old + uv_depth_av
        else:
            uv = solution
            uv_old = solution_old

        f = -(Dx(self.test[0], 0)*uv[0]*uv_old[0]
              + Dx(self.test[0], 1)*uv[0]*uv_old[1]
              + Dx(self.test[1], 0)*uv[1]*uv_old[0]
              + Dx(self.test[1], 1)*uv[1]*uv_old[1])*self.dx
        uv_av = avg(uv_old)
        un_av = (uv_av[0]*self.normal('-')[0]
                 + uv_av[1]*self.normal('-')[1])
        s = 0.5*(sign(un_av) + 1.0)
        uv_up = uv('-')*s + uv('+')*(1-s)
        if self.horizontal_continuity in ['dg', 'hdiv']:
            f += (uv_up[0]*uv_av[0]*jump(self.test[0], self.normal[0])
                  + uv_up[0]*uv_av[1]*jump(self.test[0], self.normal[1])
                  + uv_up[1]*uv_av[0]*jump(self.test[1], self.normal[0])
                  + uv_up[1]*uv_av[1]*jump(self.test[1], self.normal[1]))*(self.dS_v + self.dS_h)
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*(jump(self.test[0])*jump(uv[0])
                            + jump(self.test[1])*jump(uv[1]))*(self.dS_v + self.dS_h)
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if funcs is None:
                    un = dot(uv, self.normal)
                    uv_ext = uv - 2*un*self.normal
                    if self.use_lax_friedrichs:
                        gamma = 0.5*abs(un)*lax_friedrichs_factor
                        f += gamma*(self.test[0]*(uv[0] - uv_ext[0])
                                    + self.test[1]*(uv[1] - uv_ext[1]))*ds_bnd
                else:
                    uv_in = uv
                    if 'symm' in funcs:
                        # use internal normal velocity
                        # NOTE should this be symmetric normal velocity?
                        uv_ext = uv_in
                    elif 'uv' in funcs:
                        # prescribe external velocity
                        uv_ext = funcs['uv']
                        un_ext = dot(uv_ext, self.normal)
                    elif 'un' in funcs:
                        # prescribe normal velocity
                        un_ext = funcs['un']
                        uv_ext = self.normal*un_ext
                    elif 'flux' in funcs:
                        # prescribe normal volume flux
                        sect_len = Constant(self.boundary_len[bnd_marker])
                        eta = fields_old['eta']
                        total_h = self.bathymetry + eta
                        un_ext = funcs['flux'] / total_h / sect_len
                        uv_ext = self.normal*un_ext
                    else:
                        raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
                    if self.use_nonlinear_equations:
                        # add interior flux
                        f += (uv_in[0]*self.test[0]*self.normal[0]*uv_in[0]
                              + uv_in[0]*self.test[0]*self.normal[1]*uv_in[1]
                              + uv_in[1]*self.test[1]*self.normal[0]*uv_in[0]
                              + uv_in[1]*self.test[1]*self.normal[1]*uv_in[1])*ds_bnd
                        # add boundary contribution if inflow
                        uv_av = 0.5*(uv_in + uv_ext)
                        un_av = uv_av[0]*self.normal[0] + uv_av[1]*self.normal[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        f += (1-s)*((uv_ext - uv_in)[0]*self.test[0]*self.normal[0]*uv_av[0]
                                    + (uv_ext - uv_in)[0]*self.test[0]*self.normal[1]*uv_av[1]
                                    + (uv_ext - uv_in)[1]*self.test[1]*self.normal[0]*uv_av[0]
                                    + (uv_ext - uv_in)[1]*self.test[1]*self.normal[1]*uv_av[1])*ds_bnd

        # surf/bottom boundary conditions: closed at bed, symmetric at surf
        f += (uv_old[0]*uv[0]*self.test[0]*self.normal[0]
              + uv_old[0]*uv[1]*self.test[0]*self.normal[1]
              + uv_old[1]*uv[0]*self.test[1]*self.normal[0]
              + uv_old[1]*uv[1]*self.test[1]*self.normal[1])*(self.ds_surf)
        return -f


class VerticalAdvectionTerm(MomentumTerm):
    r"""
    Vertical advection term, :math:`\partial \left(w\textbf{u} \right)/(\partial z)`

    The weak form reads

    .. math::
        \int_\Omega \frac{\partial \left(w\textbf{u} \right)}{\partial z} \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \left( w \textbf{u} \right) \cdot \frac{\partial \boldsymbol{\psi}}{\partial z} dx
        + \int_{\mathcal{I}_{h}} \textbf{u}^{\text{up}} \cdot \text{jump}(\boldsymbol{\psi} n_z) \text{avg}(w) dS
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        w = fields_old.get('w')
        w_mesh = fields_old.get('w_mesh')
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')
        if w is None or not self.use_nonlinear_equations:
            return 0
        f = 0

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
        else:
            uv = solution

        vertvelo = w[2]
        if w_mesh is not None:
            vertvelo = w[2]-w_mesh
        adv_v = -(Dx(self.test[0], 2)*uv[0]*vertvelo
                  + Dx(self.test[1], 2)*uv[1]*vertvelo)
        f += adv_v * self.dx
        if self.vertical_continuity in ['dg', 'hdiv']:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
            uv_up = uv('-')*s + uv('+')*(1-s)
            f += (uv_up[0]*w_av*jump(self.test[0], self.normal[2])
                  + uv_up[1]*w_av*jump(self.test[1], self.normal[2]))*self.dS_h
            if self.use_lax_friedrichs:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                f += gamma*(jump(self.test[0])*jump(uv[0])
                            + jump(self.test[1])*jump(uv[1]))*self.dS_h
        f += (uv[0]*vertvelo*self.test[0]*self.normal[2]
              + uv[1]*vertvelo*self.test[1]*self.normal[2])*(self.ds_surf)
        # NOTE bottom impermeability condition is naturally satisfied by the defition of w
        return -f


class HorizontalViscosityTerm(MomentumTerm):
    r"""
    Horizontal viscosity term, :math:`- \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right) \cdot \boldsymbol{\psi} dx
        =& -\int_\Omega \nu_h (\nabla_h \boldsymbol{\psi}) : (\nabla_h \textbf{u})^T dx \\
        &+ \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\boldsymbol{\psi} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \textbf{u}) dS
        + \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \boldsymbol{\psi}) dS \\
        &- \int_{\mathcal{I}_h \cup \mathcal{I}_v} \sigma \text{avg}(\nu_h) \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{jump}(\boldsymbol{\psi} \textbf{n}_h) dS

    where :math:`\sigma` is a penalty parameter, see Hillewaert (2013).

    Hillewaert, Koen (2013). Development of the discontinuous Galerkin method
    for high-resolution, large scale CFD and acoustics in industrial
    geometries. PhD Thesis. Université catholique de Louvain.
    https://dial.uclouvain.be/pr/boreal/object/boreal:128254/
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_h = fields_old.get('viscosity_h')
        sipg_factor = self.sipg_factor
        if viscosity_h is None:
            return 0
        f = 0

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
        else:
            uv = solution

        def grad_h(a):
            return as_matrix([[Dx(a[0], 0), Dx(a[0], 1), 0],
                              [Dx(a[1], 0), Dx(a[1], 1), 0],
                              [0, 0, 0]])
        visc_tensor = as_matrix([[viscosity_h, 0, 0],
                                 [0, viscosity_h, 0],
                                 [0, 0, 0]])

        grad_uv = grad_h(uv)
        grad_test = grad_h(self.test)
        stress = dot(visc_tensor, grad_uv)
        f += inner(grad_test, stress)*self.dx

        if self.horizontal_continuity in ['dg', 'hdiv']:
            h_cell = self.mesh.ufl_cell().sub_cells()[0]
            p, q = self.function_space.ufl_element().degree()
            cp = (p + 1) * (p + 2) / 2 if h_cell == triangle else (p + 1)**2
            # by default the factor is multiplied by 2 to ensure convergence
            sigma = cp * FacetArea(self.mesh) / CellVolume(self.mesh)
            sp = sigma('+')
            sm = sigma('-')
            sigma_max = sipg_factor * conditional(sp > sm, sp, sm)
            ds_interior = (self.dS_h + self.dS_v)
            f += sigma_max*inner(
                tensor_jump(self.normal, self.test),
                dot(avg(visc_tensor), tensor_jump(self.normal, uv))
            )*ds_interior
            f += -inner(avg(dot(visc_tensor, nabla_grad(self.test))),
                        tensor_jump(self.normal, uv))*ds_interior
            f += -inner(tensor_jump(self.normal, self.test),
                        avg(dot(visc_tensor, nabla_grad(uv))))*ds_interior

        # symmetric bottom boundary condition
        f += -inner(stress, outer(self.test, self.normal))*self.ds_surf
        f += -inner(stress, outer(self.test, self.normal))*self.ds_bottom

        # TODO boundary conditions
        # TODO impermeability condition at bottom
        # TODO implement as separate function
        return -f


class VerticalViscosityTerm(MomentumTerm):
    r"""
    Vertical viscosity term, :math:`- \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right) \cdot \boldsymbol{\psi} dx
        =& -\int_\Omega \nu \frac{\partial \boldsymbol{\psi}}{\partial z} \cdot \frac{\partial \textbf{u}}{\partial z} dx \\
        &+ \int_{\mathcal{I}_h} \text{jump}(\boldsymbol{\psi} n_z) \cdot \text{avg}\left(\nu \frac{\partial \textbf{u}}{\partial z}\right) dS
        + \int_{\mathcal{I}_h} \text{jump}(\textbf{u} n_z) \cdot \text{avg}\left(\nu \frac{\partial \boldsymbol{\psi}}{\partial z}\right) dS \\
        &- \int_{\mathcal{I}_h} \sigma \text{avg}(\nu) \text{jump}(\textbf{u} n_z) \cdot \text{jump}(\boldsymbol{\psi} n_z) dS

    where :math:`\sigma` is a penalty parameter, see Hillewaert (2013).

    Hillewaert, Koen (2013). Development of the discontinuous Galerkin method
    for high-resolution, large scale CFD and acoustics in industrial
    geometries. PhD Thesis. Université catholique de Louvain.
    https://dial.uclouvain.be/pr/boreal/object/boreal:128254/
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_v = fields_old.get('viscosity_v')
        sipg_factor = self.sipg_factor_vertical
        if viscosity_v is None:
            return 0
        f = 0
        grad_test = Dx(self.test, 2)
        diff_flux = viscosity_v*Dx(solution, 2)
        f += inner(grad_test, diff_flux)*self.dx

        if self.vertical_continuity in ['dg', 'hdiv']:
            p, q = self.function_space.ufl_element().degree()
            cp = (q + 1)**2
            l_normal = CellVolume(self.mesh) / FacetArea(self.mesh)
            # by default the factor is multiplied by 2 to ensure convergence
            sigma = sipg_factor * cp / l_normal
            sp = sigma('+')
            sm = sigma('-')
            sigma_max = conditional(sp > sm, sp, sm)
            ds_interior = (self.dS_h)
            f += sigma_max*inner(
                tensor_jump(self.normal[2], self.test),
                avg(viscosity_v)*tensor_jump(self.normal[2], solution)
            )*ds_interior
            f += -inner(avg(viscosity_v*Dx(self.test, 2)),
                        tensor_jump(self.normal[2], solution))*ds_interior
            f += -inner(tensor_jump(self.normal[2], self.test),
                        avg(viscosity_v*Dx(solution, 2)))*ds_interior
        return -f


class BottomFrictionTerm(MomentumTerm):
    r"""
    Quadratic bottom friction term, :math:`\tau_b = C_D \| \textbf{u}_b \| \textbf{u}_b`

    The weak formulation reads

    .. math::
        \int_{\Gamma_{bot}} \tau_b \cdot \boldsymbol{\psi} dx = \int_{\Gamma_{bot}} C_D \| \textbf{u}_b \| \textbf{u}_b \cdot \boldsymbol{\psi} dx

    where :math:`\textbf{u}_b` is reconstructed velocity in the middle of the
    bottom element:

    .. math::
        \textbf{u}_b = \textbf{u}\Big|_{\Gamma_{bot}} + \frac{\partial \textbf{u}}{\partial z}\Big|_{\Gamma_{bot}} h_b,

    :math:`h_b` being half of the element height.
    For implicit solvers we linearize the stress as
    :math:`\tau_b = C_D \| \textbf{u}_b^{n} \| \textbf{u}_b^{n+1}`

    The drag is computed from the law-of-the wall

    .. math::
        C_D = \left( \frac{\kappa}{\ln (h_b + z_0)/z_0} \right)^2

    where :math:`z_0` is the bottom roughness length field.
    The user can override the :math:`C_D` value by providing ``quadratic_drag_coefficient``
    field.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if self.use_bottom_friction:
            uv_depth_av = fields_old.get('uv_depth_av')
            if uv_depth_av is not None:
                uv = solution + uv_depth_av
                uv_old = solution_old + uv_depth_av
            else:
                uv = solution
                uv_old = solution_old

            z_bot = Constant(0.5)*self.v_elem_size
            drag = fields_old.get('quadratic_drag_coefficient')
            if drag is None:
                bfr_roughness = fields_old.get('bottom_roughness')
                assert bfr_roughness is not None, \
                    'if use_bottom_friction=True, either bottom_roughness or quadratic_drag_coefficient must be defined'
                von_karman = physical_constants['von_karman']
                drag = (von_karman / ln((z_bot + bfr_roughness)/bfr_roughness))**2
            # compute uv_bottom implicitly
            uv_bot_old = uv_old + Dx(uv_old, 2)*z_bot
            uv_bot_mag = sqrt(uv_bot_old[0]**2 + uv_bot_old[1]**2)
            stress = drag*uv_bot_mag**2
            # Scale bottom stress by uv_{n+1}/|uv_{n} + eps| at bottom
            uv_old_mag = sqrt(uv_old[0]**2 + uv_old[1]**2)
            epsilon = Constant(1e-3)
            bfr = stress / (uv_old_mag + epsilon)
            f += bfr * (self.test[0]*uv[0] + self.test[1]*uv[1])*self.ds_bottom
        return -f


class LinearDragTerm(MomentumTerm):
    r"""
    Linear drag term, :math:`\tau_b = D \textbf{u}_b`

    where :math:`D` is the drag coefficient, read from ``linear_drag_coefficient`` field.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        # Linear drag (consistent with drag in 2D mode)
        if linear_drag_coefficient is not None:
            uv_depth_av = fields_old.get('uv_depth_av')
            if uv_depth_av is not None:
                uv = solution + uv_depth_av
            else:
                uv = solution
            bottom_fri = linear_drag_coefficient*inner(self.test, uv)*self.dx
            f += bottom_fri
        return -f


class CoriolisTerm(MomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-solution[1]*self.test[0]
                           + solution[0]*self.test[1])*self.dx
        return -f


class SourceTerm(MomentumTerm):
    r"""
    Generic momentum source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \cdot \boldsymbol{\psi} dx

    where :math:`\sigma` is a user defined vector valued :class:`Function`.

    This term also implements the wind stress, :math:`-\tau_w/(H \rho_0)`.
    :math:`\tau_w` is a user-defined wind stress :class:`Function`
    ``wind_stress``. The weak form is

    .. math::
        F_w = \int_{\Gamma_s} \frac{1}{\rho_0} \tau_w \cdot \boldsymbol{\psi} dx

    Wind stress is only included if vertical viscosity is provided.
    """
    # TODO implement wind stress as a separate term
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        viscosity_v = fields_old.get('viscosity_v')
        wind_stress = fields_old.get('wind_stress')
        if wind_stress is not None and viscosity_v is None:
            warning('Wind stress is prescribed but vertical viscosity is not:\n  Wind stress will be ignored.')
        if viscosity_v is not None:
            # wind stress
            if wind_stress is not None:
                f -= (wind_stress[0]*self.test[0]
                      + wind_stress[1]*self.test[1])/rho_0*self.ds_surf
        if source is not None:
            f += - inner(source, self.test)*self.dx
        return -f


class MomentumEquation(Equation):
    """
    Hydrostatic 3D momentum equation :eq:`mom_eq_split` for mode split models
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_nonlinear_equations=True, use_lax_friedrichs=True,
                 use_bottom_friction=False, sipg_factor=Constant(1.0),
                 sipg_factor_vertical=Constant(1.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_nonlinear_equations: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        :kwarg sipg_factor: :class: `Constant` or :class: `Function` horizontal SIPG penalty scaling factor
        :kwarg sipg_factor_vertical: :class: `Constant` or :class: `Function` vertical SIPG penalty scaling factor
        """
        # TODO rename for reflect the fact that this is eq for the split eqns
        super(MomentumEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_nonlinear_equations,
                use_lax_friedrichs, use_bottom_friction,
                sipg_factor, sipg_factor_vertical)
        self.add_term(PressureGradientTerm(*args), 'source')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(VerticalViscosityTerm(*args), 'explicit')
        self.add_term(BottomFrictionTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')


class InternalPressureGradientCalculator(MomentumTerm):
    r"""
    Computes the internal pressure gradient term, :math:`g\nabla_h r`

    where :math:`r` is the baroclinic head :eq:`baroc_head`.

    If :math:`r` belongs to a discontinuous function space, the term is
    integrated by parts:

    .. math::
        \int_\Omega g \nabla_h r \cdot \boldsymbol{\psi} dx
            = - \int_\Omega g r \nabla_h \cdot \boldsymbol{\psi} dx
            + \int_{\mathcal{I}_h \cup \mathcal{I}_v} g \text{avg}(r) \text{jump}(\boldsymbol{\psi} \cdot \textbf{n}_h) dx

    .. note ::
        Due to the :class:`Term` sign convention this term is assembled on the right-hand-side.
    """
    def __init__(self, fields, bathymetry, bnd_functions,
                 internal_pg_scalar=None, solver_parameters=None):
        """
        :arg solver: `class`FlowSolver` object
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        self.fields = fields
        self.internal_pg_scalar = internal_pg_scalar
        function_space = self.fields.int_pg_3d.function_space()
        super(InternalPressureGradientCalculator, self).__init__(
            function_space, bathymetry=bathymetry)

        solution = self.fields.int_pg_3d
        fields = {
            'baroc_head': self.fields.baroc_head_3d,
        }
        l = -self.residual(solution, solution, fields, fields,
                           bnd_conditions=bnd_functions)
        trial = TrialFunction(self.function_space)
        a = inner(trial, self.test) * self.dx
        prob = LinearVariationalProblem(a, l, solution)
        self.lin_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """
        Computes internal pressure gradient and stores it in int_pg_3d field
        """
        self.lin_solver.solve()

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        bhead = fields_old.get('baroc_head')

        if bhead is None:
            return 0

        by_parts = element_continuity(bhead.function_space().ufl_element()).horizontal == 'dg'

        if by_parts:
            div_test = (Dx(self.test[0], 0) + Dx(self.test[1], 1))
            f = -g_grav*bhead*div_test*self.dx
            head_star = avg(bhead)
            jump_n_dot_test = (jump(self.test[0], self.normal[0])
                               + jump(self.test[1], self.normal[1]))
            f += g_grav*head_star*jump_n_dot_test*(self.dS_v + self.dS_h)
            n_dot_test = (self.normal[0]*self.test[0]
                          + self.normal[1]*self.test[1])
            f += g_grav*bhead*n_dot_test*(self.ds_bottom + self.ds_surf)
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if bhead is not None:
                    if funcs is not None and 'baroc_head' in funcs:
                        r_ext = funcs['baroc_head']
                        head_ext = r_ext
                        head_in = bhead
                        head_star = 0.5*(head_ext + head_in)
                    else:
                        head_star = bhead
                    f += g_grav*head_star*n_dot_test*ds_bnd

        else:
            grad_head_dot_test = (Dx(bhead, 0)*self.test[0]
                                  + Dx(bhead, 1)*self.test[1])
            f = g_grav * grad_head_dot_test * self.dx
        if self.internal_pg_scalar is not None:
            f = self.internal_pg_scalar*f

        return -f
