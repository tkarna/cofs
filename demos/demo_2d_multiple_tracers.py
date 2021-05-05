# 2D setup with muliple tracers
# =============================
#
# The problem setup of this demo is almost identical to
# `demo_2d_tracer`. The main difference is that it shows
# how to treat the three advected quantities as separate
# tracer fields.

from thetis import *

mesh2d = UnitSquareMesh(40, 40, quadrilateral=True)
P1_2d = FunctionSpace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d)
bathymetry2d.assign(1.0)

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)

# Again, tracer functionality is activated by setting the
# option ``solve_tracer = True`` and for this problem we
# also set ``tracer_only = True``. In `demo_2d_tracer`, a
# single tracer field was used, with the default name
# `tracer_2d`. To specify separate tracers, we need to
# provide short and long form names for each, as well
# as any source terms and boundary conditions. ::

tracers = ['bell', 'cone', 'slot_cyl']
names = ['Gaussian bell', 'Cone', 'Slotted cylinder']
options = solver_obj.options
options.solve_tracer = True
options.tracer_only = True
options.fields_to_export = tracers
bc = {'value': {'on_boundary': Constant(1.0)}}
for tracer, name in zip(tracers, names):
    options.add_tracer_2d(name, tracer, source=None)
    solver_obj.bnd_functions[tracer] = bc

# Most of the remaining model setup is as before.

options.timestepper_type = 'SSPRK33'
options.timestep = pi/300.0
options.simulation_end_time = 2*pi
options.simulation_export_time = pi/15.0
options.timestepper_options.use_automatic_timestep = False
options.use_lax_friedrichs_tracer = False
options.horizontal_diffusivity = None
options.use_limiter_for_tracers = False

vP1_2d = VectorFunctionSpace(mesh2d, "CG", 1)
x, y = SpatialCoordinate(mesh2d)
uv_init = interpolate(as_vector([0.5 - y, x - 0.5]), vP1_2d)

# Initial conditions for each tracer are defined as before,
# but must be assigned separately. ::

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

bell_init = interpolate(1.0 + bell, P1_2d)
cone_init = interpolate(1.0 + cone, P1_2d)
slot_cyl_init = interpolate(1.0 + slot_cyl, P1_2d)
solver_obj.assign_initial_conditions(
    uv=uv_init, bell=bell_init, cone=cone_init, slot_cyl=slot_cyl_init
)

# Finally, we solve the tracer transport problem and display
# the normalised :math:`L^2` error. ::

solver_obj.iterate()
for tracer, init in zip(tracers, [bell_init, cone_init, slot_cyl_init]):
    q = solver_obj.fields[tracer]
    L2_err = sqrt(assemble((q - init)*(q - init)*dx))
    L2_init = sqrt(assemble(init*init*dx))
    print_output("Relative error {:8s}: {:.2f}%".format(tracer, L2_err/L2_init))

# This tutorial can be dowloaded as a Python script `here <demo_2d_multiple_tracers.py>`__.
