"""
Geostrophic gyre test case in 2D
================================

Stationary gyre test case. Initial condition for elevation is Gaussian bell
function. Initial velocity is obtained from analytical solution corresponding
to geostrophic balance.
"""
from thetis import *

lx = 1.0e6
nx = 20
mesh2d = RectangleMesh(nx, nx, lx, lx)
nonlin = False
depth = 1000.0
elev_amp = 3.0
t_end = 75*12*2*3600
t_export = 3600*2

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 900.
    t_end = t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
P1v_2d = get_functionspace(mesh2d, 'CG', 1, vector=True)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
x, y = SpatialCoordinate(mesh2d)
coriolis_2d = Function(P1_2d)
f0 = Constant(1.0e-4)
coriolis_2d.assign(f0)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.use_nonlinear_equations = False
options.coriolis_frequency = coriolis_2d
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.timestepper_type = 'CrankNicolson'
options.timestep = 20.0
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']

solver_obj.create_equations()
sigma = Constant(160.0e3)
elev_init = Function(solver_obj.function_spaces.H_2d)
x_0 = y_0 = lx/2
elev_expr = elev_amp*exp(-((x-x_0)**2+(y-y_0)**2)/sigma**2)
elev_init.project(elev_expr)

# initial velocity: u = -g/f deta/dy, v = g/f deta/dx
uv_init = Function(solver_obj.function_spaces.U_2d)
g_grav = physical_constants['g_grav']
uv_init.project(as_vector((g_grav/f0*2*(y-y_0)/sigma**2*elev_expr,
                           -g_grav/f0*2*(x-x_0)/sigma**2*elev_expr)))

solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)

solver_obj.iterate()
