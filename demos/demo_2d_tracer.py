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

# To activate the tracer functionality of the 2D model, we need to
# add tracer fields. This is done using the
# :ref:`ModelOptions2d<model_options_2d>`.\ :py:attr:`.add_tracer_2d`
# method.
#
# The 'label' identifies the field inside Thetis.
# It should not contain spaces and typically ends with '_2d'
# for 2D problems. The 'name' exists for users to identify
# the field and may contain spaces. It will appear in the
# colourbar of any vtk outputs. Finally, the 'filename'
# is used when storing outputs, so cannot contain spaces.
# The usual Thetis convention is to use CamelCase with a
# trailing '2d'.
#
# Source terms and diffusivity coefficients should also be provided
# through this interface. We have a pure advection problem with no
# diffusivity or source terms. However, such terms can be specified
# by replacing the ``None`` values below. ::

options = solver_obj.options
labels = ['tracer_2d']
names = ['Depth averaged tracer']
filenames = ['Tracer2d']
options.fields_to_export = labels
for label, name, filename in zip(labels, names, filenames):
    options.add_tracer_2d(label, name, filename, source=None, diffusivity=None)

# As mentioned above, we are only solving the tracer equation, which
# can be specified by setting ``tracer_only = True``.

options.tracer_only = True

# We will run for time :math:`2\pi` -- a full rotation -- using a
# strong stability preserving third order Runge-Kutta method (SSPRK33).
# For consistency with the Firedrake demo, Thetis' automatic timestep
# computation functionality is switched off and the simulation time is
# split into 600 steps, giving a timestep close to the CFL limit. ::

t_end = 2*pi
timestep = pi/300.0
options.tracer_timestepper_type = 'SSPRK33'
options.timestep = timestep
options.simulation_end_time = t_end
options.simulation_export_time = pi/15.0
options.tracer_timestepper_options.use_automatic_timestep = False

# For consistency with the Firedrake demo, we do not use stabilization or slope
# limiters, both of which are used by default in Thetis. Slope limiters are used
# to obtain non-oscillatory solutions. ::

options.use_lax_friedrichs_tracer = False
options.use_limiter_for_tracers = False

# The background tracer value is imposed as an upwind inflow condition.
# In general, this would be a ``Function``, but here we just use a ``Constant``
# value. ::

solver_obj.bnd_functions['tracer_2d'] = {'on_boundary': {'value': Constant(1.0)}}

# The velocity field is set up using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh2d)
uv_init = as_vector([0.5 - y, x - 0.5])

# Now, we set up the cosine-bell--cone--slotted-cylinder initial condition. The
# first four lines declare various parameters relating to the positions of these
# objects, while the analytic expressions appear in the last three lines. This
# code is simply copied from the Firedrake version of the demo. ::

bell_r0, bell_x0, bell_y0 = 0.15, 0.25, 0.5
cone_r0, cone_x0, cone_y0 = 0.15, 0.5, 0.25
cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.5, 0.75
slot_left, slot_right, slot_top = 0.475, 0.525, 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
slot_cyl = conditional(
    sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
    conditional(And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0),
    0.0
)

# We then declare the inital condition, ``q_init``, to be the sum of these fields.
# Furthermore, we add 1 to this, so that the initial field lies between 1 and 2,
# rather than between 0 and 1.  This ensures that we can't get away with
# neglecting the inflow boundary condition.  We also save the initial state so
# that we can check the :math:`L^2`-norm error at the end. ::

q_init = Function(P1_2d).interpolate(1.0 + bell + cone + slot_cyl)
solver_obj.assign_initial_conditions(uv=uv_init, tracer_2d=q_init)

# Now we are in a position to run the time loop. We can either automatically run
# the time loop to the end time as in :doc:`2D channel example <demo_2d_channel.py>`
# or call a time iterator to explicitly run the time loop, by issuing ::

thetis_timestepper = solver_obj.create_iterator()
t_Thetis = 0

while t_Thetis < t_end - timestep:
    t_Thetis = next(thetis_timestepper)

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
