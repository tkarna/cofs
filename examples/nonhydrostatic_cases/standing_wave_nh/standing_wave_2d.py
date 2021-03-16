"""
Standing wave test case
=======================

Solves a standing wave using wave equation with non-hydrostatic pressure.

Initial condition for elevation corresponds to a standing wave.

This example tests dispersion of surface waves.
"""
from thetis import *

lx = 20.
ly = 2.
nx = 10
ny = 1
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 8
elev_amp = 0.1

outputdir = 'outputs_standing_wave_2d'

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
dt = 0.1
t_export = 0.1
t_end = 20.

# choose if using non-hydrostatic model
solve_nonhydrostatic_pressure = True

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-cg'
options.polynomial_degree = 1
# time stepper
options.timestepper_type = 'PressureProjectionPicard'
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = True
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'q_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'q_2d']
# non-hydrostatic
if solve_nonhydrostatic_pressure:
    options_nh = options.nh_model_options
    options_nh.solve_nonhydrostatic_pressure = solve_nonhydrostatic_pressure
    options_nh.q_degree = 2
    options_nh.update_free_surface = True
    options_nh.free_surface_timestepper_type = 'CrankNicolson'

# create equations
solver_obj.create_equations()

# set initial elevation
x = SpatialCoordinate(mesh2d)
elev_init = elev_amp*cos(2*pi*x[0]/lx)
solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()

# error show
anal_elev = elev_amp*cos(2*pi*x[0]/lx)*cos(sqrt(9.81*2*pi/lx*tanh(2*pi*depth/lx))*t_end)
L2_elev = errornorm(anal_elev, solver_obj.fields.elev_2d)/sqrt(lx*ly)
print('L2 error for surface elevation is ', L2_elev)
