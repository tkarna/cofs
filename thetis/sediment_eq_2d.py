r"""
2D advection diffusion equation for sediment transport.

This can be either conservative :math:`q=HT` or non-conservative :math:`T` sediment and allows
for a separate source and sink term. The equation reads

.. math::
    \frac{\partial S}{\partial t}
    + \nabla_h \cdot (\textbf{u} S)
    = \nabla_h \cdot (\mu_h \nabla_h S) + F_{source} - (F_{sink} S)
    :label: sediment_eq_2d

where :math:`S` is :math:`q` for conservative and :math:`T` for non-conservative,
:math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and :math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from .equation import Equation
from .tracer_eq_2d import HorizontalDiffusionTerm, HorizontalAdvectionTerm, TracerTerm
from .conservative_tracer_eq_2d import ConservativeHorizontalAdvectionTerm

__all__ = [
    'SedimentEquation2D',
    'SedimentTerm',
    'ConservativeSedimentAdvectionTerm',
    'SedimentAdvectionTerm',
    'SedimentErosionTerm',
    'SedimentDepositionTerm',
]


class SedimentTerm(TracerTerm):
    """
    Generic sediment term that provides commonly used members.
    """
    def __init__(self, function_space, depth, options, sediment_model, conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class:`DepthExpression` containing depth info
        :arg options: :class`ModelOptions2d` containing parameters
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentTerm, self).__init__(function_space, depth, options)
        self.sediment_model = sediment_model
        self.conservative = conservative

    def get_bnd_functions(self, c_in, uv_in, elev_in, bnd_id, bnd_conditions):
        funcs = bnd_conditions.get(bnd_id)
        c_ext, uv_ext, elev_ext = super().get_bnd_functions(c_in, uv_in, elev_in, bnd_id, bnd_conditions)
        if 'equilibrium' in funcs:
            if 'value' in funcs:
                raise ValueError("Cannot specify both equilibrium and value for sediment bcs.")
            c_ext = self.sediment_model.get_equilibrium_tracer()
            if self.conservative:
                c_ext = c_ext * self.depth.get_total_depth(elev_ext)
        return c_ext, uv_ext, elev_ext


class ConservativeSedimentAdvectionTerm(SedimentTerm, ConservativeHorizontalAdvectionTerm):
    """
    Advection term for sediment equation

    Same as :class:`ConservativeHorizontalAdvectionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass


class SedimentAdvectionTerm(SedimentTerm, HorizontalAdvectionTerm):
    """
    Advection term for sediment equation

    Same as :class:`HorizontalAdvectionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass


class SedimentDiffusionTerm(SedimentTerm, HorizontalDiffusionTerm):
    """
    Diffusion term for sediment equation

    Same as :class:`HorizontalDiffusionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass


class SedimentErosionTerm(SedimentTerm):
    """
    Erosion term for sediment equation"""
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        ero = self.sediment_model.get_erosion_term()
        if not self.conservative:
            elev = fields['elev_2d']
            ero = ero / self.depth.get_total_depth(elev)
        f = self.test * ero * self.dx
        return f


class SedimentDepositionTerm(SedimentTerm):
    """
    Deposition term for sediment equation"""
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        depo = self.sediment_model.get_deposition_coefficient()
        elev = fields['elev_2d']
        H = self.depth.get_total_depth(elev)
        f = -self.test * depo/H * solution * self.dx
        return f


class SedimentEquation2D(Equation):
    """
    2D sediment advection-diffusion equation: :eq:`tracer_eq_2d` or :eq:`conservative_tracer_eq_2d`
    with sediment source and sink term
    """
    def __init__(self, function_space, depth, options, sediment_model, conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class:`DepthExpression` containing depth info
        :arg options: :class`ModelOptions2d` containing parameters
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentEquation2D, self).__init__(function_space)
        args = (function_space, depth, options, sediment_model, conservative)
        if conservative:
            self.add_term(ConservativeSedimentAdvectionTerm(*args), 'explicit')
        else:
            self.add_term(SedimentAdvectionTerm(*args), 'explicit')
        self.add_term(SedimentDiffusionTerm(*args), 'explicit')
        self.add_term(SedimentErosionTerm(*args), 'source')
        self.add_term(SedimentDepositionTerm(*args), 'implicit')
