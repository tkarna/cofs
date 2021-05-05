"""
Idealized estuary test case
===========================

Tidal flow in a rectangual channel with a density gradient.
Setup according to [1].

Bathymetry varies between 10 m (ocean boundary) and 5 m (river boundary).
At the ocean boundary tidal flux is prescribed, while a constant influx is
used at the river boundary.
Initial salinity field is a linear ramp from 32 psu (at x=30 km) to 0 psu
(at x=80 km).
Temperature is fixed to 10 deg Celcius.
This corresponds to density 1023.05 kg/m3 in the ocean and 999.70 kg/m3
in the river.

[1] Warner, J. C., Sherwood, C. R., Arango, H. G., and Signell, R. P.
    (2005). Performance of four turbulence closure models implemented
    using a generic length scale method. Ocean Modelling, 8(1-2):81-113.
"""
from thetis import *

# set physical constants
physical_constants['rho0'].assign(1000.0)

reso_str = 'coarse'
refinement = {'coarse': 1, 'normal': 2}
lx = 100.0e3
ly = 1000.0/refinement[reso_str]
nx = int(round(100*refinement[reso_str]))
delta_x = lx/nx
ny = 2
layers = int(round(10*refinement[reso_str]))
mesh2d = RectangleMesh(nx, ny, lx, ly)
t_end = 18*24*3600
# export every 9 min, day 16 is export 2720
t_export = 9*60.0

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    layers = 5
    t_end = t_export

depth_ocean = 10
u_tide = 0.4
t_tide = 12*3600
salt_ocean = 30.0
depth_river = 5
u_river = -0.08
salt_river = 0.0
temp_const = 10.0

# bathymetry
p1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(p1_2d, name='Bathymetry')
x, y = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(depth_ocean - (depth_ocean - depth_river)*x/lx)

simple_barotropic = False  # for testing flux boundary conditions

# create solver
solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solverobj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = not simple_barotropic
options.solve_temperature = False
options.constant_temperature = Constant(temp_const)
options.use_implicit_vertical_diffusion = not simple_barotropic
options.use_bottom_friction = not simple_barotropic
options.bottom_roughness = Constant(0.005)
options.use_turbulence = not simple_barotropic
options.use_turbulence_advection = not simple_barotropic
options.use_baroclinic_formulation = not simple_barotropic
options.use_lax_friedrichs_velocity = True
options.use_lax_friedrichs_tracer = True
# options.horizontal_diffusivity = Constant(50.0)
# options.horizontal_viscosity = Constant(50.0)
options.vertical_viscosity = Constant(1.3e-6)  # background value
options.vertical_diffusivity = Constant(1.4e-7)  # background value
options.use_limiter_for_tracers = True
Re_h = 10.0
uscale = 1.0
nu_scale = uscale * delta_x / Re_h
print_output('Horizontal viscosity {:}'.format(nu_scale))
options.horizontal_viscosity = Constant(nu_scale)
options.horizontal_diffusivity = Constant(5.0)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = Constant(2.0)
options.horizontal_viscosity_scale = Constant(nu_scale)
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'salt_3d', 'density_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d']
options.fields_to_export_hdf5 = []
turbulence_model_options = options.turbulence_model_options
turbulence_model_options.apply_defaults('k-epsilon')
turbulence_model_options.stability_function_name = 'Canuto A'
outputdir = 'outputs'
odir = '_'.join([outputdir, reso_str,
                 turbulence_model_options.closure_name.replace(' ', '-'),
                 turbulence_model_options.stability_function_name.replace(' ', '-')])
options.output_directory = odir
print_output('Exporting to ' + options.output_directory)

solverobj.create_function_spaces()

# initial conditions
salt_init3d = Function(solverobj.function_spaces.H, name='initial salinity')
# original vertically uniform initial condition
x, y, z = SpatialCoordinate(solverobj.mesh)
salt_init3d.interpolate(salt_ocean - (salt_ocean - salt_river)*(x - 30e3)/50e3)
# start from idealized salt wedge
# salt_init3d.interpolate(salt_river + (salt_river - salt_ocean)*(x - 80e3)/50e3 * (0.5 - 0.5*tanh(4*(z + 2.0))))
min_ix = salt_init3d.dat.data < salt_river
salt_init3d.dat.data[min_ix] = salt_river
max_ix = salt_init3d.dat.data > salt_ocean
salt_init3d.dat.data[max_ix] = salt_ocean

# weak boundary conditions
flux_ocean = -u_tide*depth_ocean*ly
flux_river = u_river*depth_river*ly

t = 0.0
t_ramp = 3600.0  # NOTE use ramp to avoid stading waves
ocean_flux_func = lambda t: (flux_ocean*sin(2 * pi * t / t_tide)
                             - flux_river)*min(t/t_ramp, 1.0)
ocean_flux = Constant(ocean_flux_func(t))
river_flux_func = lambda t: flux_river*min(t/t_ramp, 1.0)
river_flux = Constant(river_flux_func(t))

ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}
ocean_funcs_3d = {'symm': None}
river_funcs_3d = {'symm': None}
ocean_salt_3d = {'value': salt_init3d}
river_salt_3d = {'value': salt_init3d}
solverobj.bnd_functions['shallow_water'] = {1: ocean_funcs, 2: river_funcs}
solverobj.bnd_functions['momentum'] = {1: ocean_funcs_3d, 2: river_funcs_3d}
solverobj.bnd_functions['salt'] = {1: ocean_salt_3d, 2: river_salt_3d}


def update_forcings(t_new):
    ocean_flux.assign(ocean_flux_func(t_new))
    river_flux.assign(river_flux_func(t_new))


solverobj.assign_initial_conditions(salt=salt_init3d)
solverobj.iterate(update_forcings=update_forcings)
