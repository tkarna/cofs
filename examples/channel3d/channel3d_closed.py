"""
Idealised channel flow in 3D
============================

Solves shallow water equations in closed rectangular domain
with sloping bathymetry.

Initially water elevation is set to a piecewise linear function
with a slope in the deeper (left) end of the domain. This results
in a wave that develops a shock as it reaches shallower end of the domain.
This example tests the integrity of the coupled 2D-3D model and stability
of momentum advection.

This test is also useful for testing tracer conservation and consistency
by advecting a constant passive tracer.
"""
from thetis import *

n_layers = 6
outputdir = 'outputs_closed'
lx = 100e3
ly = 3000.
nx = 80
ny = 3
mesh2d = RectangleMesh(nx, ny, lx, ly)
print_output('Exporting to ' + outputdir)
t_end = 6 * 3600
t_export = 900.0

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 900.
    t_end = t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

depth_max = 20.0
depth_min = 7.0
xy = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(depth_max - (depth_max-depth_min)*xy[0]/lx)
u_max = 4.5
w_max = 5e-3

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = True
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.use_limiter_for_tracers = True
options.use_lax_friedrichs_velocity = False
options.use_lax_friedrichs_tracer = False
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(u_max)
options.vertical_velocity_scale = Constant(w_max)
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.check_salinity_conservation = True
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d']

# initial elevation, piecewise linear function
elev_init_2d = Function(P1_2d, name='elev_2d_init')
max_elev = 6.0
elev_slope_x = 30e3
elev_init_2d.interpolate(conditional(xy[0] < elev_slope_x, -xy[0]*max_elev/elev_slope_x + max_elev, 0.0))
salt_init_3d = Constant(4.5)

solver_obj.assign_initial_conditions(elev=elev_init_2d, salt=salt_init_3d)
solver_obj.iterate()
