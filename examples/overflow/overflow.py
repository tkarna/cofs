"""
Overflow Test case
=======================

Overflow test case from Haidvogel and Beckmann (1999)

200 km long channel with sloping bathymetry from 200 m to 4 km depth.
Initially dense water is located on top of the slope.

Horizontal resolution: 1 km
Vertical layers: 40, 66, or 100 (Ilicak, 2012)
Baroclinic/barotropic time steps: 10.0 s / 1.0 s

Dianeutral mixing depends on mesh Reynolds number (Ilicak et al. 2012)
Re_h = U dx / nu
U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(g_h drho/rho_0)
dx = horizontal mesh size
nu = background viscosity
"""
from thetis import *

physical_constants['rho0'] = 999.7

reso_str = 'coarse'
refinement = {'medium': 4, 'coarse': 1}
lx = 200.0e3
delta_x = 4000./refinement[reso_str]
nx = int(lx/delta_x)
ny = 2
ly = ny*delta_x
mesh2d = RectangleMesh(nx, ny, lx, ly)
layers = 10 if reso_str == 'coarse' else 25

dt = 20.0/refinement[reso_str]
t_end = 25 * 3600
t_export = 15*60.0
depth = 20.0
Re_h = 10.0
outputdir = 'outputs_' + reso_str + '_Re' + str(int(Re_h))

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = dt
    t_end = t_export
    layers = 3

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
hmin = 500.0
hmax = 2000.0
Ls = 10.0e3
x0 = 40.0e3
x, y = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(hmin + 0.5*(hmax - hmin)*(1 + tanh((x - x0)/Ls)))

# temperature and salinity, results in 2.0 kg/m3 density difference
salt_left = 2.5489
salt_right = 0.0
temp_const = 10.0

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = True
options.solve_temperature = False
options.constant_temperature = Constant(temp_const)
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.use_baroclinic_formulation = True
options.use_lax_friedrichs_velocity = True
options.use_lax_friedrichs_tracer = True
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(Re_h))
options.use_limiter_for_tracers = True
options.vertical_viscosity = Constant(1.0e-4)
options.horizontal_viscosity = None
options.horizontal_diffusivity = None
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(6.0)
options.vertical_velocity_scale = Constant(3.0)
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.check_salinity_conservation = True
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d']

solver_obj.create_equations()
salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
x, y, z = SpatialCoordinate(solver_obj.mesh)
x0 = 20.0e3
# vertical barrier
salt_init3d.interpolate(conditional(le(x, x0), salt_left, salt_right))
# smooth condition
# sigma = 1000.0
# salt_init3d.interpolate(salt_left + (salt_right - salt_left)*0.5*(1.0 + tanh((x - x0)/sigma)))

solver_obj.assign_initial_conditions(salt=salt_init3d)
solver_obj.iterate()
