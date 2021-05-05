"""
Tracer box in 3D
================

Solves a standing wave in a rectangular basin using wave equation.

This version uses the ALE moving mesh and a constant tracer to check
tracer local/global tracer conservation.

Initial condition for elevation corresponds to a standing wave.
Time step and export interval are chosen based on theorethical
oscillation frequency. Initial condition repeats every 20 exports.
"""
from thetis import *

lx = 44294.46
ly = 2000.0
nx = 25
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 30.0
elev_amp = 2.0
n_layers = 12
# estimate of max advective velocity for computing time step
u_mag = Constant(3.0)
w_mag = Constant(2.0e-2)
sloped = True
warped = False

suffix = ''
if sloped:
    suffix = '_sloped'
if warped:
    suffix = '_warped'
outputdir = 'outputs' + suffix

print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

if sloped:
    xy = SpatialCoordinate(mesh2d)
    bathymetry_2d.interpolate(depth + 15.0*2*(xy[0]/lx - 0.5))

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
t_export = dt
t_export = 100.0
t_end = 10*T_cycle + 1e-3

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = t_export

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)

if warped:
    # create non-uniform vertical layers
    coords = solver_obj.mesh.coordinates
    z = coords.dat.data[:, 2].copy()
    x = coords.dat.data[:, 0]
    p = 1.5*x/lx + 0.01  # ~0.0 => ~unform mesh
    # p = np.ones_like(x)*0.001
    sigma = -depth * (0.5*np.tanh(p*(-2.0*z/depth - 1.0))/np.tanh(p) + 0.5)
    coords.dat.data[:, 2] = sigma

options = solver_obj.options
options.use_nonlinear_equations = True
# options.element_family = 'rt-dg'
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = True
options.solve_temperature = True
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.use_limiter_for_tracers = True
options.use_lax_friedrichs_velocity = False
options.use_lax_friedrichs_tracer = False
# options.horizontal_viscosity = Constant(100.0)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.vertical_velocity_scale = w_mag
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.check_salinity_conservation = True
options.check_salinity_overshoot = True
options.check_temperature_conservation = True
options.check_temperature_overshoot = True
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                            'uv_dav_2d']

# need to call creator to create the function spaces
solver_obj.create_equations()
elev_init = Function(solver_obj.function_spaces.H_2d)
xy = SpatialCoordinate(solver_obj.mesh2d)
elev_init.project(-elev_amp*cos(2*pi*xy[0]/lx))
x_0 = 30.0e3
ss = 0.5*(sign(xy[0] - x_0) + 1.0)
elev_init.project(5.0*ss*(xy[0] - x_0)/(lx - x_0))

salt_init3d = None
temp_init3d = None
if options.solve_salinity:
    # constant tracer field to test consistency with 3d continuity eq
    salt_init3d = Constant(4.5)
if options.solve_temperature:
    temp_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
    xyz = SpatialCoordinate(solver_obj.mesh)
    temp_l = 0
    temp_r = 30.0
    temp_init3d.interpolate(temp_l + (temp_r - temp_l)*0.5*(1.0 + sign(xyz[0] - lx/2)))
    # temp_init3d.interpolate(temp_l + (temp_r - temp_l)*0.5*(1.0 - sign(xyz[2] + 0.33*depth)))

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, temp=temp_init3d)
solver_obj.iterate()
