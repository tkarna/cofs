"""
Columbia river plume simulation
===============================

requirements:
netCDF4
pyproj
scipy
uptide

"""
from thetis import *
from bathymetry import get_bathymetry, smooth_bathymetry, smooth_bathymetry_at_bnd
from tidal_forcing import TidalBoundaryForcing
from diagnostics import TimeSeriesCallback2D
from roms_forcing import LiveOceanInterpolator
from atm_forcing import *
comm = COMM_WORLD

# TODO add intial condition from ROMS/HYCOM

rho0 = 1000.0
# set physical constants
physical_constants['rho0'].assign(rho0)
physical_constants['z0_friction'].assign(0.005)

reso_str = 'coarse'
meshfile = {
    'coarse': 'mesh_cre-plume002.msh',
    'normal': 'mesh_cre-plume003.msh',
}
zgrid_params = {
    # nlayers, surf_elem_height, max_z_stretch
    'coarse': (9, 5.0, 4.0),
    'normal': (15, 0.8, 4.0),
}
nlayers, surf_elem_height, max_z_stretch = zgrid_params[reso_str]
outputdir = 'outputs_{:}'.format(reso_str)
mesh2d = Mesh(meshfile[reso_str])
print_output('Loaded mesh ' + mesh2d.name)

nnodes = comm.allreduce(mesh2d.topology.num_vertices(), MPI.SUM)
ntriangles = comm.allreduce(mesh2d.topology.num_cells(), MPI.SUM)
nprisms = ntriangles*nlayers

sim_tz = timezone.FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)

t_end = 10*24*3600.
t_export = 900.

# interpolate bathymetry and smooth it
bathymetry_2d = get_bathymetry('bathymetry_300m.npz', mesh2d, project=False)
bathymetry_2d = smooth_bathymetry(
    bathymetry_2d, delta_sigma=1.0, bg_diff=0,
    alpha=5e6, exponent=1,
    minimum_depth=3.5, niter=20)
bathymetry_2d = smooth_bathymetry_at_bnd(bathymetry_2d, [1, 3])

# 3d mesh vertical stretch factor
z_stretch_fact_2d = Function(bathymetry_2d.function_space(), name='z_stretch')
# 1.0 (sigma mesh) in shallow areas, 4.0 in deep ocean
z_stretch_fact_2d.project(-ln(surf_elem_height/bathymetry_2d)/ln(nlayers))
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data < 1.0] = 1.0
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data > max_z_stretch] = max_z_stretch

coriolis_f, coriolis_beta = beta_plane_coriolis_params(46.25)
q_river = 5000.
salt_river = 0.0
salt_ocean_surface = 32.0
salt_ocean_bottom = 34.0
temp_river = 15.0
temp_ocean_surface = 13.0
temp_ocean_bottom = 8.0
reynolds_number = 160.0

u_scale = 3.0
w_scale = 1e-3
delta_x = 2e3
nu_scale = u_scale * delta_x / reynolds_number

simple_barotropic = False  # for debugging

# create solver
extrude_options = {
    'z_stretch_fact': z_stretch_fact_2d,
}

solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers,
                               extrude_options=extrude_options)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = not simple_barotropic
options.solve_temperature = not simple_barotropic
options.use_implicit_vertical_diffusion = True  # not simple_barotropic
options.use_bottom_friction = True  # not simple_barotropic
options.use_turbulence = True  # not simple_barotropic
options.use_turbulence_advection = False  # not simple_barotropic
options.use_smooth_eddy_viscosity = False
options.turbulence_model_type = 'pacanowski'
options.use_baroclinic_formulation = not simple_barotropic
options.use_lax_friedrichs_velocity = True
options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
options.use_lax_friedrichs_tracer = True
options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
options.vertical_viscosity = Constant(2e-5)
options.vertical_diffusivity = Constant(2e-5)
options.horizontal_viscosity = Constant(2.0)
options.horizontal_diffusivity = Constant(2.0)
options.use_quadratic_pressure = True
options.use_limiter_for_tracers = True
options.use_limiter_for_velocity = True
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(reynolds_number))
options.coriolis_frequency = Constant(coriolis_f)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(u_scale)
options.vertical_velocity_scale = Constant(w_scale)
options.horizontal_viscosity_scale = Constant(nu_scale)
options.check_salinity_overshoot = True
options.check_temperature_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'density_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d',
                            'int_pg_3d', 'hcc_metric_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
options.equation_of_state_type = 'full'
# gls_options = options.turbulence_model_options
# gls_options.apply_defaults('k-omega')
# gls_options.stability_name = 'CB'
options.turbulence_model_options.alpha = 10.
options.turbulence_model_options.exponent = 2
options.turbulence_model_options.max_viscosity = 0.05

solver_obj.create_function_spaces()

# atm forcing
wind_stress_3d = Function(solver_obj.function_spaces.P1v, name='wind stress')
wind_stress_2d = Function(solver_obj.function_spaces.P1v_2d, name='wind stress')
atm_pressure_2d = Function(solver_obj.function_spaces.P1_2d, name='atm pressure')
options.wind_stress = wind_stress_3d
options.atmospheric_pressure = atm_pressure_2d
copy_wind_stress_to_3d = ExpandFunctionTo3d(wind_stress_2d, wind_stress_3d)
wrf_pattern = 'forcings/atm/wrf/wrf_air.{year}_*_*.nc'.format(year=init_date.year)
wrf_atm = WRFInterpolator(
    solver_obj.function_spaces.P1_2d,
    wind_stress_2d, atm_pressure_2d, wrf_pattern, init_date)
wrf_atm.set_fields(0.0)

# ocean initial conditions
salt_init_3d = Function(solver_obj.function_spaces.P1, name='initial salinity')
temp_init_3d = Function(solver_obj.function_spaces.P1, name='initial temperature')
liveocean_interp = LiveOceanInterpolator(solver_obj.function_spaces.P1,
                                         [salt_init_3d, temp_init_3d],
                                         ['salt', 'temp'],
                                         'forcings/liveocean/f2015.*/ocean_his_*.nc',
                                         init_date)
liveocean_interp.set_fields(0.0)


# tides
elev_tide_2d = Function(bathymetry_2d.function_space(), name='Boundary elevation')
bnd_time = Constant(0)

ramp_t = 12*3600.
elev_ramp = conditional(le(bnd_time, ramp_t), bnd_time/ramp_t, 1.0)
elev_bnd_expr = elev_ramp*elev_tide_2d

north_bnd_id = 1
west_bnd_id = 2
south_bnd_id = 3
river_bnd_id = 4

bnd_elev_updater = TidalBoundaryForcing(
    elev_tide_2d, init_date,
    boundary_ids=[north_bnd_id, west_bnd_id, south_bnd_id])

# river flux
river_flux_interp = interpolation.NetCDFTimeSeriesInterpolator(
    'forcings/stations/bvao3/bvao3.0.A.FLUX/*.nc',
    ['flux'], init_date, scalars=[-1.0])
river_flux_const = Constant(river_flux_interp(0)[0])

river_swe_funcs = {'flux': river_flux_const}
tide_elev_funcs = {'elev': elev_bnd_expr}
zero_elev_funcs = {'elev': Constant(0)}
open_uv_funcs = {'symm': None}
zero_uv_funcs = {'uv': Constant((0, 0, 0))}
bnd_river_salt = {'value': Constant(salt_river)}
ocean_salt_funcs = {'value': salt_init_3d}
bnd_river_temp = {'value': Constant(temp_river)}
ocean_temp_funcs = {'value': temp_init_3d}
solver_obj.bnd_functions['shallow_water'] = {
    river_bnd_id: river_swe_funcs,
    south_bnd_id: tide_elev_funcs,
    north_bnd_id: tide_elev_funcs,
    west_bnd_id: tide_elev_funcs,
}
solver_obj.bnd_functions['momentum'] = {
    river_bnd_id: open_uv_funcs,
    south_bnd_id: zero_uv_funcs,
    north_bnd_id: zero_uv_funcs,
    west_bnd_id: zero_uv_funcs,
}
solver_obj.bnd_functions['salt'] = {
    river_bnd_id: bnd_river_salt,
    south_bnd_id: ocean_salt_funcs,
    north_bnd_id: ocean_salt_funcs,
    west_bnd_id: ocean_salt_funcs,
}
solver_obj.bnd_functions['temp'] = {
    river_bnd_id: bnd_river_temp,
    south_bnd_id: ocean_temp_funcs,
    north_bnd_id: ocean_temp_funcs,
    west_bnd_id: ocean_temp_funcs,
}

solver_obj.create_equations()

solver_obj.add_callback(
    TimeSeriesCallback2D(
        solver_obj, 'elev_2d', x=357450.0, y=287571.0, location_name='tpoin'))

hcc_obj = Mesh3DConsistencyCalculator(solver_obj)
hcc_obj.solve()

print_output('Running CRE plume with options:')
print_output('Resolution: {:}'.format(reso_str))
print_output('Reynolds number: {:}'.format(reynolds_number))
print_output('Horizontal viscosity: {:}'.format(nu_scale))
print_output('Exporting to {:}'.format(outputdir))

solver_obj.assign_initial_conditions(salt=salt_init_3d, temp=temp_init_3d)


# def show_uv_mag():
#     uv = solver_obj.fields.uv_3d.dat.data
#     print_output('uv: {:9.2e} .. {:9.2e}'.format(uv.min(), uv.max()))
#     ipg = solver_obj.fields.int_pg_3d.dat.data
#     print_output('int pg: {:9.2e} .. {:9.2e}'.format(ipg.min(), ipg.max()))


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_updater.set_tidal_field(t)
    river_flux_const.assign(river_flux_interp(t)[0])
    liveocean_interp.set_fields(t)
    wrf_atm.set_fields(t)
    copy_wind_stress_to_3d.solve()


out_atm_pressure = File(options.outputdir + '/AtmPressure2d.pvd')
out_wind_stress = File(options.outputdir + '/WindStress2d.pvd')


def export_atm_fields():
    out_atm_pressure.write(atm_pressure_2d)
    out_wind_stress.write(wind_stress_2d)


solver_obj.iterate(update_forcings=update_forcings, export_func=export_atm_fields)
