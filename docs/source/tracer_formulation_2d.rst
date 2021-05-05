2D tracer formulation
=====================

.. highlight:: python

Governing equation
------------------

The two dimensional tracer model solves an advection-diffusion
equation
:eq:`tracer_eq_2d`.
If solved in non-conservative form, the prognostic variable
is the passive tracer concentration,
:math:`T`. The corresponding field in Thetis is called
``'tracer_2d'``.

A conservative tracer model is also available, given by
:eq:`cons_tracer_eq_2d`.
In this case, the equation is solved for :math:`q=HT`, where
:math:`H` is the total water depth.
The conservative tracer model is specified using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_tracer_conservative_form`
option.

To activate the 2D tracer model, set the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.solve_tracer`
option to
``True``. The tracer model may also be run independently
by setting the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.tracer_only`
option to
``True``. The hydrodynamics will be defined by any initial
conditions specified for the horizontal velocity, or any updates
imposed by the user.

Spatial discretization
----------------------

Thetis supports two different tracer finite element discretizations
and associated stabilization methods, summarised in the table below.

=============== ========= ======= ===============
Element Family  Name      Space   Stabilization
=============== ========= ======= ===============
DG              ``'dg'``  P1DG    Lax-Friedrichs
CG              ``'cg'``  P1      SUPG
=============== ========= ======= ===============

Table 1. *Finite element families and stabilization methods.*

The element family is set by the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.tracer_element_family`
option. Polynomial degrees other than one are not currently supported.

Lax-Friedrichs stabilization is used by default and may be
controlled using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_lax_friedrichs_tracer`
option. Note that it is only a valid choice for the ``'dg'`` element family.
The scaling parameter used by this scheme may be controlled using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.lax_friedrichs_tracer_scaling_factor`
option.

If the ``'cg'`` element family is chosen, then SUPG stabilization is used by
default. It can be controlled using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_supg_tracer`
option. In that case, it is advisable to set characteristic velocities and
diffusivities for your problem using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.horizontal_velocity_scale`
and
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.horizontal_diffusivity_scale`
options.

Temporal discretization
-----------------------

Thetis supports different time integration methods, set by the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.timestepper_type` option.
Note that the same time integration method will be used for both the shallow
water equations and the 2D tracer model.

==================== ============================ ====================== ============
Time integrator      Thetis class                 Unconditionally stable Description
==================== ============================ ====================== ============
``'ForwardEuler'``   :py:class:`~.ForwardEuler`   No                     Forward Euler method
``'BackwardEuler'``  :py:class:`~.BackwardEuler`  Yes                    Backward Euler method
``'CrankNicolson'``  :py:class:`~.CrankNicolson`  Yes                    Crank-Nicolson method
``'DIRK22'``         :py:class:`~.DIRK22`         Yes                    DIRK(2,3,2) method
``'DIRK33'``         :py:class:`~.DIRK33`         Yes                    DIRK(3,4,3) method
``'SSPRK33'``        :py:class:`~.SSPRK33`        No                     SSPRK(3,3) method
``'SteadyState'``    :py:class:`~.SteadyState`    --                     Solves equations in steady state
==================== ============================ ====================== ============

Table 2. *Time integration methods for 2D tracer model.*
