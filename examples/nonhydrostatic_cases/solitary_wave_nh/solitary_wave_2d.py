"""
Solitary wave propagation test case
===================================

Solves a solitary wave propagating in a constant-depth channel.

Initial condition for elevation and velocity fields by Boussinesq solution.

This example tests solitary wave propagation.
"""
import numpy as np
from thetis import *

lx = 1000.0
ly = 2.
nx = 500
ny = 1
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 10.0

outputdir = 'outputs_solitary_wave_2d'

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
dt = 0.1
t_export = 0.1
t_end = 50.

# choose if using non-hydrostatic model
solve_nonhydrostatic_pressure = True

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'CrankNicolson'
# time stepper
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = False
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

# --- create equations ---
solver_obj.create_equations()

# set initial elevation and velocity
g_grav = 9.81
e = 0.2  # e = H/depth
H = e*depth  # soliatry wave height
x0 = 200
c = np.sqrt(g_grav*(depth + H))
alpha = np.sqrt(3./4.*H/depth**3)

elev_init = Function(solver_obj.function_spaces.H_2d)
uv_init = Function(solver_obj.function_spaces.U_2d)
x, y = SpatialCoordinate(mesh2d)
t = 0
elev_init.interpolate(H*cosh(alpha*(x - x0 - c*t))**(-2))
uv_init.dat.data[:, 0] = np.sqrt(g_grav*depth)*elev_init.dat.data[:]/depth
solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)

solver_obj.iterate()

# error show
anal_elev = Function(solver_obj.function_spaces.H_2d)
anal_elev.interpolate(H*cosh(alpha*(x - x0 - c*t_end))**(-2))
L2_elev = errornorm(anal_elev, solver_obj.fields.elev_2d)/sqrt(lx*ly)
print('L2 error for surface elevation is ', L2_elev)
