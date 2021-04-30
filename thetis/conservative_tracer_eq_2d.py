r"""
2D advection diffusion equation for tracers.

The advection-diffusion equation of depth-integrated tracer :math:`q=HT` in conservative form reads

.. math::
    \frac{\partial q}{\partial t}
    + \nabla_h \cdot (\textbf{u} q)
    = \nabla_h \cdot (\mu_h \nabla_h q)
    :label: cons_tracer_eq_2d

where :math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and
:math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from .utility import *
from .tracer_eq_2d import HorizontalDiffusionTerm, TracerTerm, TracerEquation2D

__all__ = [
    'ConservativeTracerEquation2D',
    'ConservativeTracerTerm',
    'ConservativeHorizontalAdvectionTerm',
    'ConservativeHorizontalDiffusionTerm',
    'ConservativeSourceTerm',
]


class ConservativeTracerTerm(TracerTerm):
    """
    Generic depth-integrated tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth, options, test_function=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class`ModelOptions2d` containing parameters
        """
        super(ConservativeTracerTerm, self).__init__(function_space, depth, options,
                                                     test_function=test_function)

    # TODO: at the moment this is the same as TracerTerm, but we probably want to overload its
    # get_bnd_functions method


class ConservativeHorizontalAdvectionTerm(ConservativeTracerTerm):
    r"""
    Advection of tracer term, :math:`\nabla \cdot \bar{\textbf{u}} \nabla q`

    The weak form is

    .. math::
        \int_\Omega \boldsymbol{\psi} \nabla\cdot \bar{\textbf{u}} \nabla q  dx
        = - \int_\Omega \left(\nabla_h \boldsymbol{\psi})\right) \cdot \bar{\textbf{u}} \cdot q dx
        + \int_\Gamma \text{avg}(q\bar{\textbf{u}}\cdot\textbf{n}) \cdot \text{jump}(\boldsymbol{\psi}) dS

    where the right hand side has been integrated by parts;
    :math:`\textbf{n}` is the unit normal of
    the element interfaces, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')

        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        f += -(Dx(self.test, 0) * uv[0] * solution
               + Dx(self.test, 1) * uv[1] * solution) * self.dx

        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            flux_up = solution('-')*uv('-')*s + solution('+')*uv('+')*(1-s)

            f += (flux_up[0] * jump(self.test, self.normal[0])
                  + flux_up[1] * jump(self.test, self.normal[1])) * self.dS
            # Lax-Friedrichs stabilization
            if self.options.use_lax_friedrichs_tracer:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
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
                    flux_up = c_in*uv*s + c_ext*uv_ext*(1-s)
                    f += (flux_up[0]*self.normal[0]
                          + flux_up[1]*self.normal[1])*self.test*ds_bnd
                else:
                    f += c_in * (uv[0]*self.normal[0]
                                 + uv[1]*self.normal[1])*self.test*ds_bnd

        return -f


class ConservativeHorizontalDiffusionTerm(ConservativeTracerTerm, HorizontalDiffusionTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h q)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla_h \cdot (\mu_h \nabla_h q) \phi dx
        =& \int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h q) dx \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h q) dS
        - \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(q \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(q \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS \\
        &- \int_\Gamma \mu_h (\nabla_h \phi) \cdot \textbf{n}_h ds

    where :math:`\sigma` is a penalty parameter, see Hillewaert (2013).

    Hillewaert, Koen (2013). Development of the discontinuous Galerkin method
    for high-resolution, large scale CFD and acoustics in industrial
    geometries. PhD Thesis. Université catholique de Louvain.
    https://dial.uclouvain.be/pr/boreal/object/boreal:128254/
    """
    # TODO: at the moment the same as HorizontalDiffusionTerm
    # do we need additional H-derivative term?
    # would also become different if ConservativeTracerTerm gets different bc options


class ConservativeSourceTerm(ConservativeTracerTerm):
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
            H = self.depth.get_total_depth(fields_old['elev_2d'])
            f += -inner(H*source, self.test)*self.dx
        return -f


class ConservativeTracerEquation2D(TracerEquation2D):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def add_terms(self, *args, **kwargs):
        self.add_term(ConservativeHorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(ConservativeSourceTerm(*args, **kwargs), 'source')
