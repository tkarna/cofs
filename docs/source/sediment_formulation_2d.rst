2D sediment transport formulation
=================================

.. highlight:: python

Governing equations
-------------------

Suspended sediment transport is modelled in two dimensions
using an advection-diffusion equation
:eq:`sediment_eq_2d`.
If solved in non-conservative form, the prognostic variable
is the passive tracer concentration,
:math:`T`. The corresponding field in Thetis is called
``'sediment_2d'``.

A conservative suspended sediment transport model is also
available. In this case, the equation is solved for
:math:`q=HT`, where
:math:`H` is the total water depth.
The conservative tracer model is specified using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_sediment_conservative_form`
option.

Bedload transport is modelled in two dimensions using the
Exner equation
:eq:`exner_eq`.
It is solved for the bedlevel,
:math:`z_b`, which is the bathymetry. The corresponding field
in Thetis is called
``'bathymetry_2d'``.

To activate the 2D sediment model, set the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.solve_suspended_sediment`
and
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.solve_exner`
options to
``True``.

Spatial discretization
----------------------

Thetis currently only supports suspended sediment in P1DG space.
Lax-Friedrichs stabilization is used by default and may be
controlled using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_lax_friedrichs_tracer`
option. The scaling parameter used by this scheme may be controlled
using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.lax_friedrichs_tracer_scaling_factor`
option.

The function space used for the bedlevel is determined by that
used for the bathymetry. Typically, this is P1.

Temporal discretization
-----------------------

Thetis supports different time integration methods, set by the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.timestepper_type` option.
Note that the same time integration method will be used for both the shallow
water equations and the 2D sediment model.

=============================== ====================================== ====================== ============
Time integrator                 Thetis class                           Unconditionally stable Description
=============================== ====================================== ====================== ============
``'ForwardEuler'``              :py:class:`~.ForwardEuler`             No                     Forward Euler method
``'BackwardEuler'``             :py:class:`~.BackwardEuler`            Yes                    Backward Euler method
``'CrankNicolson'``             :py:class:`~.CrankNicolson`            Yes                    Crank-Nicolson method
``'DIRK22'``                    :py:class:`~.DIRK22`                   Yes                    DIRK(2,3,2) method
``'DIRK33'``                    :py:class:`~.DIRK33`                   Yes                    DIRK(3,4,3) method
``'SSPRK33'``                   :py:class:`~.SSPRK33`                  No                     SSPRK(3,3) method
=============================== ====================================== ====================== ============

Table 1. *Time integration methods for 2D sediment model.*
