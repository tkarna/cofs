"""
Idealised channel flow in 3D
============================

Solves shallow water equations in rectangular domain
with sloping bathymetry.

Flow is forced with tidal volume flux in the deep (ocean) end of the
channel, and a constant volume flux in the shallow (river) end.

This example demonstrates how to set up time dependent boundary conditions.
"""
from thetis import *

n_layers = 6
outputdir = 'outputs'
lx = 100e3
ly = 3000.
nx = 80
ny = 3
mesh2d = RectangleMesh(nx, ny, lx, ly)
print_output('Exporting to ' + outputdir)
t_end = 24 * 3600
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
u_max = 2.0
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
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d']

# initial conditions
salt_init3d = Constant(4.5)

# weak boundary conditions
un_amp = -0.5
flux_amp = ly*depth_max*un_amp
t_tide = 12 * 3600.
un_river = -0.3
flux_river = ly*depth_min*un_river
t = 0.0
t_ramp = 12*3600.0  # use linear ramp up for boundary forcings
# python function that returns time dependent boundary values
ocean_flux_func = lambda t: (flux_amp*sin(2 * pi * t / t_tide)
                             - flux_river)*min(t/t_ramp, 1.0)
river_flux_func = lambda t: flux_river*min(t/t_ramp, 1.0)
# Constants that will be fed to the model
ocean_flux = Constant(ocean_flux_func(t))
river_flux = Constant(river_flux_func(t))

# boundary conditions are defined with a dict
# key defines the type of bnd condition, value the necessary coefficient(s)
# here setting outward bnd flux (positive outward)
ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}
ocean_funcs_3d = {'symm': None}
river_funcs_3d = {'symm': None}
# and constant salinity (for inflow)
ocean_salt_3d = {'value': salt_init3d}
river_salt_3d = {'value': salt_init3d}
# bnd conditions are assigned to each boundary tag with another dict
ocean_tag = 1
river_tag = 2
# assigning conditions for each equation
# these must be assigned before equations are created
solver_obj.bnd_functions['shallow_water'] = {ocean_tag: ocean_funcs,
                                             river_tag: river_funcs}
solver_obj.bnd_functions['momentum'] = {ocean_tag: ocean_funcs_3d,
                                        river_tag: river_funcs_3d}
solver_obj.bnd_functions['salt'] = {ocean_tag: ocean_salt_3d,
                                    river_tag: river_salt_3d}


def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 2d mode"""
    ocean_flux.assign(ocean_flux_func(t_new))
    river_flux.assign(river_flux_func(t_new))


# set init conditions, this will create all function spaces, equations etc
solver_obj.assign_initial_conditions(salt=salt_init3d)
solver_obj.iterate(update_forcings=update_forcings)
