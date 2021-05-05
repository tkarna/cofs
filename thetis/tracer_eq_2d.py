r"""
2D advection diffusion equation for tracers.

The advection-diffusion equation of tracer :math:`T` in non-conservative form reads

.. math::
    \frac{\partial T}{\partial t}
    + \nabla_h \cdot (\textbf{u} T)
    = \nabla_h \cdot (\mu_h \nabla_h T)
    :label: tracer_eq_2d

where :math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and
:math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'TracerEquation2D',
    'TracerTerm',
    'HorizontalAdvectionTerm',
    'HorizontalDiffusionTerm',
    'SourceTerm',
]


class TracerTerm(Term):
    """
    Generic tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth, options, test_function=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class:`DepthExpression` containing depth info
        :arg options: :class`ModelOptions2d` containing parameters
        :kwarg test_function: custom :class:`TestFunction`.
        """
        super(TracerTerm, self).__init__(function_space,
                                         test_function=test_function)
        self.depth = depth
        self.options = options
        self.cellsize = CellSize(self.mesh)
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_dg = continuity.horizontal == 'dg'

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)

    def get_bnd_functions(self, c_in, uv_in, elev_in, bnd_id, bnd_conditions):
        """
        Returns external values of tracer and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.

        :arg c_in: Internal value of tracer
        :arg uv_in: Internal value of horizontal velocity
        :arg elev_in: Internal value of elevation
        :arg bnd_id: boundary id
        :type bnd_id: int
        :arg bnd_conditions: dict of boundary conditions:
            ``{bnd_id: {field: value, ...}, ...}``
        """
        funcs = bnd_conditions.get(bnd_id)

        if 'elev' in funcs:
            elev_ext = funcs['elev']
        else:
            elev_ext = elev_in
        if 'value' in funcs:
            c_ext = funcs['value']
        else:
            c_ext = c_in
        if 'uv' in funcs:
            uv_ext = self.corr_factor * funcs['uv']
        elif 'flux' in funcs:
            h_ext = self.depth.get_total_depth(elev_ext)
            area = h_ext*self.boundary_len[bnd_id]  # NOTE using external data only
            uv_ext = self.corr_factor * funcs['flux']/area*self.normal
        elif 'un' in funcs:
            uv_ext = funcs['un']*self.normal
        else:
            uv_ext = uv_in

        return c_ext, uv_ext, elev_ext


class HorizontalAdvectionTerm(TracerTerm):
    r"""
    Advection of tracer term, :math:`\bar{\textbf{u}} \cdot \nabla T`

    The weak form is

    .. math::
        \int_\Omega \bar{\textbf{u}} \cdot \boldsymbol{\psi} \cdot \nabla T  dx
        = - \int_\Omega \nabla_h \cdot (\bar{\textbf{u}} \boldsymbol{\psi}) \cdot T dx
        + \int_\Gamma \text{avg}(T) \cdot \text{jump}(\boldsymbol{\psi}
        (\bar{\textbf{u}}\cdot\textbf{n})) dS

    where the right hand side has been integrated by parts;
    :math:`\textbf{n}` is the unit normal of
    the element interfaces, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.

    For the continuous Galerkin method we use

    .. math::
        \int_\Omega \bar{\textbf{u}} \cdot \boldsymbol{\psi} \cdot \nabla T  dx
        = - \int_\Omega \nabla_h \cdot (\bar{\textbf{u}} \boldsymbol{\psi}) \cdot T dx.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        f += -(Dx(uv[0] * self.test, 0) * solution
               + Dx(uv[1] * self.test, 1) * solution) * self.dx

        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)

            f += c_up*(jump(self.test, uv[0] * self.normal[0])
                       + jump(self.test, uv[1] * self.normal[1])) * self.dS
            # Lax-Friedrichs stabilization
            if self.options.use_lax_friedrichs_tracer:
                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution
                if funcs is not None:
                    c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                    uv_av = 0.5*(uv + uv_ext)
                    un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    f += c_up*(uv_av[0]*self.normal[0]
                               + uv_av[1]*self.normal[1])*self.test*ds_bnd
                else:
                    f += c_in * (uv[0]*self.normal[0]
                                 + uv[1]*self.normal[1])*self.test*ds_bnd

        return -f


class HorizontalDiffusionTerm(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        =& \int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h T) dS
        - \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(T \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(T \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS \\
        &- \int_\Gamma \mu_h (\nabla_h \phi) \cdot \textbf{n}_h ds

    where :math:`\sigma` is a penalty parameter, see Hillewaert (2013).

    For the continuous Galerkin method we use
    .. math::
        -\int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        = \int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx.

    Hillewaert, Koen (2013). Development of the discontinuous Galerkin method
    for high-resolution, large scale CFD and acoustics in industrial
    geometries. PhD Thesis. Université catholique de Louvain.
    https://dial.uclouvain.be/pr/boreal/object/boreal:128254/
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])
        grad_test = grad(self.test)
        diff_flux = dot(diff_tensor, grad(solution))
        sipg_factor = self.options.sipg_factor_tracer

        f = 0
        f += inner(grad_test, diff_flux)*self.dx

        if self.horizontal_dg:
            cell = self.mesh.ufl_cell()
            p = self.function_space.ufl_element().degree()
            cp = (p + 1) * (p + 2) / 2 if cell == triangle else (p + 1)**2
            l_normal = CellVolume(self.mesh) / FacetArea(self.mesh)
            # by default the factor is multiplied by 2 to ensure convergence
            sigma = sipg_factor * cp / l_normal
            sp = sigma('+')
            sm = sigma('-')
            sigma_max = conditional(sp > sm, sp, sm)
            ds_interior = self.dS
            f += sigma_max * inner(
                jump(self.test, self.normal),
                dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
            f += -inner(avg(dot(diff_tensor, grad(self.test))),
                        jump(solution, self.normal))*ds_interior
            f += -inner(jump(self.test, self.normal),
                        avg(dot(diff_tensor, grad(solution))))*ds_interior

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution
                elev = fields_old['elev_2d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor * fields_old['uv_2d']
                if funcs is not None:
                    if 'diff_flux' in funcs:
                        f += -self.test*funcs['diff_flux']*ds_bnd
                    else:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        diff_flux_up = dot(diff_tensor, grad(c_up))
                        f += -self.test*dot(diff_flux_up, self.normal)*ds_bnd

        return -f


class SourceTerm(TracerTerm):
    r"""
    Generic source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += -inner(source, self.test)*self.dx
        return -f


class TracerEquation2D(Equation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space, depth, options, velocity):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class`ModelOptions2d` containing parameters
        :arg velocity: velocity field associated with the shallow water model
        """
        super(TracerEquation2D, self).__init__(function_space)

        # Apply SUPG stabilisation
        kwargs = {}
        if options.use_supg_tracer:
            unorm = options.horizontal_velocity_scale
            if unorm.values()[0] > 0:
                cellsize = anisotropic_cell_size(function_space.mesh())
                tau = 0.5*cellsize/unorm
                D = options.horizontal_diffusivity_scale
                if D.values()[0] > 0:
                    Pe = 0.5*unorm*cellsize/D
                    tau = min_value(tau, Pe/3)
                self.test = self.test + tau*dot(velocity, grad(self.test))
                kwargs['test_function'] = self.test

        args = (function_space, depth, options)
        self.add_term(HorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm(*args, **kwargs), 'source')
