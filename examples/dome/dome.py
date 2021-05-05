"""
DOME test case
==============

Initially the water column is linearly stratified with density difference of
2 kg/m3 in the deepest part of the domain.

In the embayment there is a geostrophically balanced inflow of 5e6 m3/s with
positive density anomaly 2 kg/m3.

Radiation boundary conditions are applied in the east and west open boundaries.

Typical mesh resolution is dx=10 km, 21 sigma levels [2]

[1] Ezer and Mellor (2004). A generalized coordinate ocean model and a
    comparison of the bottom boundary layer dynamics in terrain-following and
    in z-level grids. Ocean Modelling, 6(3-4):379-403.
    http://dx.doi.org/10.1016/S1463-5003(03)00026-X
[2] Burchard and Rennau (2008). Comparative quantification of physically and
    numerically induced mixing in ocean models. Ocean Modelling, 20(3):293-311.
    http://dx.doi.org/10.1016/j.ocemod.2007.10.003
[3] Legg et al. (2006). Comparison of entrainment in overflows simulated by
    z-coordinate, isopycnal and non-hydrostatic models. Ocean Modelling, 11(1-2):69-97.
"""
from thetis import *
import dome_setup as setup
import diagnostics

comm = COMM_WORLD

physical_constants['rho0'] = setup.rho_0

reso_str = 'coarse'
delta_x_dict = {'normal': 6e3, 'coarse': 20e3}
n_layers_dict = {'normal': 24, 'coarse': 7}
n_layers = n_layers_dict[reso_str]
mesh2d = Mesh('mesh_{0:s}.msh'.format(reso_str))
print_output('Loaded mesh '+mesh2d.name)
t_end = 47 * 24 * 3600
t_export = 3 * 3600
outputdir = 'outputs_' + reso_str

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 900.
    t_end = t_export
    n_layers = 3

delta_x = delta_x_dict[reso_str]

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
xy = SpatialCoordinate(mesh2d)
lin_bath_expr = (setup.depth_lim[1] - setup.depth_lim[0])/(setup.y_slope[1] - setup.y_slope[0])*(xy[1] - setup.y_slope[0]) + setup.depth_lim[0]
bathymetry_2d.interpolate(lin_bath_expr)
bathymetry_2d.dat.data[bathymetry_2d.dat.data > setup.depth_lim[0]] = setup.depth_lim[0]
bathymetry_2d.dat.data[bathymetry_2d.dat.data < setup.depth_lim[1]] = setup.depth_lim[1]

# estimate velocity / diffusivity scales
u_max_int = np.sqrt(setup.g/setup.rho_0*setup.delta_rho/setup.depth_lim[0])*setup.depth_lim[0]/np.pi
u_max = 3.5
w_max = 3e-2

# compute horizontal viscosity
uscale = 2.0
reynolds_number = 240.
nu_scale = uscale * delta_x / reynolds_number

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
outputdir += '_' + options.element_family

options.timestepper_type = 'SSPRK22'
options.solve_salinity = True
options.solve_temperature = True
options.use_implicit_vertical_diffusion = True
options.use_bottom_friction = True
options.use_turbulence = True
options.turbulence_model_type = 'pacanowski'
options.use_ale_moving_mesh = True
options.use_baroclinic_formulation = True
options.use_lax_friedrichs_velocity = False
options.use_lax_friedrichs_tracer = False
options.coriolis_frequency = Constant(setup.f_0)
options.use_limiter_for_tracers = True
options.use_limiter_for_velocity = True
options.use_lax_friedrichs_tracer = True
options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
options.use_lax_friedrichs_velocity = True
options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
options.quadratic_drag_coefficient = Constant(0.002)
options.vertical_viscosity = Constant(2e-5)
options.horizontal_viscosity = Constant(nu_scale)
options.horizontal_diffusivity = Constant(10.0)
options.vertical_diffusivity = Constant(2e-5)
options.use_quadratic_pressure = True
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_viscosity_scale = Constant(nu_scale)
options.horizontal_velocity_scale = Constant(u_max)
options.vertical_velocity_scale = Constant(w_max)
options.check_temperature_overshoot = True
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'temp_3d', 'salt_3d',
                            'density_3d', 'uv_dav_2d', 'uv_dav_3d',
                            'baroc_head_3d', 'smag_visc_3d',
                            'eddy_visc_3d', 'eddy_diff_3d',
                            'int_pg_3d', 'hcc_metric_3d']
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d', 'uv_3d', 'salt_3d',
                                 'temp_3d', 'tke_3d', 'psi_3d']
options.equation_of_state_type = 'linear'
options.equation_of_state_options.rho_ref = setup.rho_0
options.equation_of_state_options.s_ref = setup.salt_const
options.equation_of_state_options.th_ref = setup.temp_lim[1]
options.equation_of_state_options.alpha = setup.alpha
options.equation_of_state_options.beta = setup.beta
options.turbulence_model_options.alpha = 10.
options.turbulence_model_options.exponent = 2
options.turbulence_model_options.max_viscosity = 0.05

solver_obj.create_function_spaces()
solver_obj.create_fields()

xyz = SpatialCoordinate(solver_obj.mesh)

# create additional fields for imposing inflow boudary conditions
temp_expr = (setup.temp_lim[1] - setup.temp_lim[0])*(setup.depth_lim[0] + xyz[2])/setup.depth_lim[0] + setup.temp_lim[0]
temp_init_3d = Function(solver_obj.function_spaces.H, name='inflow temperature')
temp_init_3d.interpolate(temp_expr)
# this is inefficient! find a way to do this without allocating fields
x_arr = Function(solver_obj.function_spaces.H).interpolate(xyz[0]).dat.data[:]
y_arr = Function(solver_obj.function_spaces.H).interpolate(xyz[1]).dat.data[:]
z_arr = Function(solver_obj.function_spaces.H).interpolate(xyz[2]).dat.data[:]
x_w_arr = x_arr - setup.bay_x_lim[0]
ix = y_arr > setup.basin_ly + 50e3  # assign only in the bay
temp_init_3d.dat.data[ix] = setup.temp_func(x_w_arr[ix], z_arr[ix])

# add a relaxation term for temperature:
# dT/dt ... - 1/tau*(T_relax - T) = 0
temp_relax = temp_init_3d
t_temp_relax = Constant(6.*3600.)  # time scale
mask_temp_relax_3d = Function(solver_obj.function_spaces.H, name='mask_temp_relax_3d')
lx_relax = 160e3
mask_numpy_x0 = (1 - (x_arr + setup.basin_extend)/lx_relax)
mask_numpy_x1 = (x_arr-setup.basin_lx)/lx_relax + 1
mask_temp_relax_3d.dat.data[:] = np.maximum(mask_numpy_x0, mask_numpy_x1)
ix = mask_temp_relax_3d.dat.data < 0
mask_temp_relax_3d.dat.data[ix] = 0.0
# File('mask.pvd').write(mask_temp_relax_3d)
options.temperature_source_3d = mask_temp_relax_3d/t_temp_relax*(temp_relax - solver_obj.fields.temp_3d)

# use salinity field as a passive tracer for tracking inflowing waters
salt_init_3d = Function(solver_obj.function_spaces.H, name='inflow salinity')
# mark waters T < 15.0 degC as 1.0, 0.0 otherwise
ix = y_arr > setup.basin_ly + 50e3  # assign only in the bay
salt_init_3d.dat.data[ix] = (setup.temp_lim[1] - setup.temp_func(x_w_arr[ix], z_arr[ix]))/(setup.temp_lim[1] - setup.temp_lim[0])

uv_inflow_3d = Function(solver_obj.function_spaces.P1DGv, name='inflow velocity')
uv_inflow_3d.dat.data[ix, 1] = setup.v_func(x_w_arr[ix], z_arr[ix])
uv_inflow_2d = Function(solver_obj.function_spaces.P1DGv_2d, name='inflow velocity')

# compute total volume flux at inflow bnd
init_inflow = abs(assemble(dot(uv_inflow_3d, FacetNormal(solver_obj.mesh))*ds_v(int(4))))
target_inflow = 5e6
flow_corr_fact = Constant(target_inflow/init_inflow)
tot_inflow = abs(assemble(dot(flow_corr_fact*uv_inflow_3d, FacetNormal(solver_obj.mesh))*ds_v(int(4))))

bhead_init_3d = Function(solver_obj.fields.baroc_head_3d.function_space(), name='init bhead')


def compute_depth_av_inflow(uv_inflow_3d, uv_inflow_2d):
    """Computes depth average of 3d field. Should only be called once."""
    tmp_inflow_3d = Function(solver_obj.function_spaces.P1DGv)
    inflow_averager = VerticalIntegrator(uv_inflow_3d,
                                         tmp_inflow_3d,
                                         bottom_to_top=True,
                                         bnd_value=Constant((0.0, 0.0, 0.0)),
                                         average=True,
                                         bathymetry=solver_obj.fields.bathymetry_2d.view_3d,
                                         elevation=solver_obj.fields.elev_cg_2d.view_3d)
    inflow_extract = SubFunctionExtractor(tmp_inflow_3d,
                                          uv_inflow_2d,
                                          boundary='top', elem_facet='top')
    inflow_averager.solve()
    inflow_extract.solve()
    # remove depth av. from 3D
    uv_inflow_3d += -tmp_inflow_3d


# set boundary conditions
radiation_swe_bnd = {'elev': Constant(0.0), 'uv': Constant((0, 0))}
outflow_swe_bnd = {'elev': Constant(0.0), 'flux': Constant(tot_inflow)}
inflow_swe_bnd = {'uv': flow_corr_fact*uv_inflow_2d}
inflow_salt_bnd = {'value': salt_init_3d}
inflow_temp_bnd = {'value': temp_init_3d}
zero_salt_bnd = {'value': Constant(0.0)}
inflow_uv_bnd = {'uv': flow_corr_fact*uv_inflow_3d, 'baroc_head': bhead_init_3d}
outflow_uv_bnd = {'flux': Constant(tot_inflow), 'baroc_head': bhead_init_3d}
zero_uv_bnd = {'uv': Constant((0, 0, 0)), 'baroc_head': bhead_init_3d}

bnd_id_west = 1
bnd_id_east = 2
bnd_id_inflow = 4
solver_obj.bnd_functions['shallow_water'] = {
    bnd_id_inflow: inflow_swe_bnd,
    bnd_id_west: outflow_swe_bnd,
    bnd_id_east: radiation_swe_bnd,
}
solver_obj.bnd_functions['momentum'] = {
    bnd_id_inflow: inflow_uv_bnd,
    bnd_id_west: outflow_uv_bnd,
    bnd_id_east: zero_uv_bnd,
}
solver_obj.bnd_functions['temp'] = {
    bnd_id_inflow: inflow_temp_bnd,
    bnd_id_west: inflow_temp_bnd,
    bnd_id_east: inflow_temp_bnd,
}
solver_obj.bnd_functions['salt'] = {
    bnd_id_inflow: inflow_salt_bnd,
    bnd_id_west: zero_salt_bnd,
    bnd_id_east: zero_salt_bnd,
}

solver_obj.create_equations()

solver_obj.add_callback(
    diagnostics.VerticalProfileCallback(
        solver_obj, 'salt_3d', x=700e3, y=560e3, npoints=48))
solver_obj.add_callback(
    diagnostics.TracerHistogramCallback(
        solver_obj, 'salt_3d', x_bins=np.linspace(0, 850e3, 61), rho_bins=np.linspace(0.0, 2.0, 41)))

compute_depth_av_inflow(uv_inflow_3d, uv_inflow_2d)
tot_inflow_2d = abs(assemble(dot(setup.depth_lim[1]*flow_corr_fact*uv_inflow_2d, FacetNormal(solver_obj.mesh2d))*ds(int(4))))

hcc_obj = Mesh3DConsistencyCalculator(solver_obj)
hcc_obj.solve()

print_output('Running DOME problem with options:')
print_output('Resolution: {:}'.format(reso_str))
print_output('Reynolds number: {:}'.format(reynolds_number))
hcc = solver_obj.fields.hcc_metric_3d.dat.data
print_output('HCC mesh consistency: {:} .. {:}'.format(hcc.min(), hcc.max()))
print_output('Horizontal viscosity: {:}'.format(nu_scale))
print_output('Internal wave speed: {:.3f}'.format(u_max_int))
print_output('Total inflow: {:.3f} Sv (uncorrected {:.3f} Sv)'.format(tot_inflow/1e6, init_inflow/1e6))
print_output('Total 2D inflow: {:.3f} Sv'.format(tot_inflow_2d/1e6))
print_output('Exporting to {:}'.format(outputdir))


# Export bottom salinity
bot_salt_2d = Function(solver_obj.function_spaces.H_2d, name='Salinity')
extract_bot_salt = SubFunctionExtractor(solver_obj.fields.salt_3d, bot_salt_2d, boundary='bottom')
bot_salt_file = File(options.output_directory + '/BotSalinity2d.pvd')


def export_func():
    extract_bot_salt.solve()
    bot_salt_file.write(bot_salt_2d)


solver_obj.assign_initial_conditions(temp=temp_init_3d, salt=salt_init_3d)
# use initial baroclinic head on the boundary
bhead_init_3d.assign(solver_obj.fields.baroc_head_3d)
solver_obj.iterate(export_func=export_func)
