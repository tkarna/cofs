# 2D tracer transport
# ===================
#
# This demo shows how the Firedrake DG advection equation
# `demo <https://firedrakeproject.org/demos/DG_advection.py.html>`__
# can be implemented in Thetis.
#
# The test case is the classic cosine-bell--cone--slotted-cylinder
# advection test case of :cite:`LeVeque:1996`. The domain is the unit
# square :math:`\Omega=[0,1]^2` and the velocity corresponds to the
# solid body rotation :math:`\vec{u} = (0.5 - y, x - 0.5)`.
# 
# As usual, we start by importing Thetis. ::

from thetis import *

# Define a 40-by-40 mesh of squares. ::

mesh2d = UnitSquareMesh(40, 40, quadrilateral=True)

# We will solve a pure advection problem in non-conservative form,
# with no hydrodynamics. Therefore, bathymetry is not actually
# important. We set an arbitrary postive value, as this is required
# by Thetis to construct the solver object. ::

P1_2d = FunctionSpace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d)
bathymetry2d.assign(1.0)
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)

# To activate the tracer functionality, we set the option 
# ``solve_tracer = True``. As mentioned above, we are only solving
# the tracer equation, which can be specified by setting
# ``tracer_only = True``.

options = solver_obj.options
options.solve_tracer = True
options.tracer_only = True
options.fields_to_export = ['tracer_2d']

# We will run for time :math:`2\pi` -- a full rotation -- using a
# strong stability preserving third order Runge-Kutta method (SSPRK33).
# For consistency with the Firedrake demo, Thetis' automatic timestep
# computation functionality is switched off and the simulation time is
# split into 600 steps, giving a timestep close to the CFL limit. ::

options.timestepper_type = 'SSPRK33'
options.timestep = pi/300.0
options.simulation_end_time = 2*pi
options.simulation_export_time = pi/15.0
options.timestepper_options.use_automatic_timestep = False

# We have a pure advection problem with no diffusivity or source terms. However,
# such terms can be specified by replacing the ``None`` values below. For
# consistency with the Firedrake demo, we do not use stabilization or slope
# limiters, both of which are used by default in Thetis. Slope limiters are used
# to obtain non-oscillatory solutions. ::

options.horizontal_diffusivity = None
options.tracer_source_2d = None
options.use_lax_friedrichs_tracer = False
options.use_limiter_for_tracers = False

# The background tracer value is imposed as an upwind inflow condition.
# In general, this would be a ``Function``, but here we just use a ``Constant``
# value. ::

solver_obj.bnd_functions['tracer'] = {'on_boundary':  {'value': Constant(1.0)}}

# The velocity field is set up using a simple analytic expression. ::

vP1_2d = VectorFunctionSpace(mesh2d, "CG", 1)
x, y = SpatialCoordinate(mesh2d)
uv_init = interpolate(as_vector([0.5 - y, x - 0.5]), vP1_2d)

# Now, we set up the cosine-bell--cone--slotted-cylinder initial condition. The
# first four lines declare various parameters relating to the positions of these
# objects, while the analytic expressions appear in the last three lines. This
# code is simply copied from the Firedrake version of the demo. ::

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

# We then declare the inital condition, ``q_init``, to be the sum of these fields.
# Furthermore, we add 1 to this, so that the initial field lies between 1 and 2,
# rather than between 0 and 1.  This ensures that we can't get away with
# neglecting the inflow boundary condition.  We also save the initial state so
# that we can check the :math:`L^2`-norm error at the end. ::

q_init = interpolate(1.0 + bell + cone + slot_cyl, P1_2d)
solver_obj.assign_initial_conditions(uv=uv_init, tracer=q_init)

# Now we are in a position to run the time loop. ::

solver_obj.iterate()

# Finally, we display the normalised :math:`L^2` error, by comparing to the initial condition. ::

q = solver_obj.fields.tracer_2d
L2_err = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init = sqrt(assemble(q_init*q_init*dx))
print_output(L2_err/L2_init)

# This tutorial can be dowloaded as a Python script `here <demo_2d_tracer.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
