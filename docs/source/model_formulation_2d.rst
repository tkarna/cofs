Depth averaged 2D model formulation
===================================

.. highlight:: python

Governing equations
-------------------

The two dimensional model solves the depth averaged shallow water equations
:eq:`swe_freesurf`\-:eq:`swe_momentum`.
The prognostic variables are the water elevation :math:`\eta` and depth
averaged velocity :math:`\bar{\mathbf{u}}`.
The corresponding fields in Thetis are called ``'elev_2d'`` and  ``'uv_2d'``.

Wetting and drying
------------------

Wetting and drying is included through the modified bathymetry formulation of
Karna et al. (2011). The modified equations are given by
:eq:`swe_freesurf_wd`\-:eq:`swe_momentum_wd`. The :math:`\alpha` parameter
associated with the wetting and drying scheme can be specified by the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.wetting_and_drying_alpha`
option. It can also be computed automatically, by setting the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_automatic_wetting_and_drying_alpha`
option to ``True``.

Spatial discretization
----------------------

Thetis supports different finite element discretizations, summarised in the
table below.

.. |uu| replace:: :math:`\bar{\mathbf{u}}`
.. |eta| replace:: :math:`\eta`

======================== ============ =========== ========== ===========
Element Family           Name         Degree *n*  |uu| space |eta| space
======================== ============ =========== ========== ===========
Equal order DG           ``'dg-dg'``  1, 2        P(n)DG     P(n)DG
Raviart-Thomas DG        ``'rt-dg'``  1, 2        RT(n+1)    P(n)DG
P1DG-P2                  ``'dg-cg'``  1           P(n)DG     P(n+1)
Brezzi-Douglas-Marini DG ``'bdm-dg'`` 1, 2        BDM(n+1)   P(n)DG
======================== ============ =========== ========== ===========

Table 1. *Finite element families for polynomial degree n.*

The element family and polynomial degree are set by the :ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.element_family` and :ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.polynomial_degree` options.

Lax-Friedrichs stabilization is used by default and may be controlled using
the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.use_lax_friedrichs_velocity`
option. The scaling parameter used by this scheme may be controlled using the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.lax_friedrichs_velocity_scaling_factor`
option.

Temporal discretization
-----------------------

Thetis supports different time integration methods, set by the
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.timestepper_type` option.

=============================== ====================================== ====================== ============
Time integrator                 Thetis class                           Unconditionally stable Description
=============================== ====================================== ====================== ============
``'ForwardEuler'``              :py:class:`~.ForwardEuler`             No                     Forward Euler method
``'BackwardEuler'``             :py:class:`~.BackwardEuler`            Yes                    Backward Euler method
``'CrankNicolson'``             :py:class:`~.CrankNicolson`            Yes                    Crank-Nicolson method
``'DIRK22'``                    :py:class:`~.DIRK22`                   Yes                    DIRK(2,3,2) method
``'DIRK33'``                    :py:class:`~.DIRK33`                   Yes                    DIRK(3,4,3) method
``'SSPRK33'``                   :py:class:`~.SSPRK33`                  No                     SSPRK(3,3) method
``'SSPIMEX'``                   :py:class:`~.IMEXLPUM2`                No                     LPUM2 SSP IMEX scheme
``'PressureProjectionPicard'``  :py:class:`~.PressureProjectionPicard` No                     Efficient pressure projection solver
``'SteadyState'``               :py:class:`~.SteadyState`              --                     Solves equations in steady state
=============================== ====================================== ====================== ============

Table 2. *Time integration methods for 2D model.*

Model time step is defined by the :ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.timestep` option.

For explicit solvers, Thetis can also estimate the maximum stable time step
based on the mesh resolution, used element family and time integration scheme.
To use this feature, the user should provide the maximal horizontal velocity
scale with :ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.horizontal_velocity_scale` option and set
:ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.timestepper_options`.\ :py:attr:`.use_automatic_timestep` to ``True``.
