"""
Baroclinic channel test case following [1]

The domain extends from latitude 30 N to 45 N and is 20 deg long in zonal
direction. Periodic boundaries are used in eastern and western boundaries.
Depth is constant 1600 m. Mesh resolution is 1/6 deg in zonal and 1/7 deg
in meridional directions. 24 evenly-spaced vertical levels are used. In [2]
resolutions 1/3, 1/6, 1/9, 1/12, 1/24, and 1/36 deg are used.

Here the domain is approximated with a 1600-by-1600 km box in Cartesian
coordinates with the following resolutions

n_elem   delta x (m)    deg  levels
-----------------------------------
 43      37209.30      ~1/3      20
 86      18604.65      ~1/6      20
 172     9302.326     ~1/12      20
 344     4651.163     ~1/24      20

Initial temperature is defined by gradients dT/dy=-5e-6 degC/m and
dT/dz=8.2e-3 degC/m. Maximum surface temperature is 25 degC.
Salinity is constant 35 psu.

Vertical viscosity and diffusivity are set to 1e-3 and 1e-5 m2/s, respectively.
Pacanowski-Philander vertical mixing scheme is used for temperature with max.
diffusivity 0.01 m2/s. Horizontal diffusivity 30 m2/s is used. Horizontal
viscosity depends on the advection scheme (typ. 100 m2/s). Bottom drag
coefficient is Cd=0.0025.

[1] Danilov (2012). Two finite-volume unstructured mesh models for
large-scale ocean modeling. Ocean Modelling, 47:14-25.
[2] Danilov and Wang (2015). Resolving eddies by local mesh refinement.
 Ocean Modelling, 93:75-83.
"""

from thetis import *
comm = COMM_WORLD

physical_constants['rho0'] = 1020.

refinement = 1  # normal = 4
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    refinement = 0.5
lx = ly = 1600e3
nx = ny = int(43*refinement)
delta_x = lx/nx
delta_y = ly/ny
mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x')
depth = 1600.
nlayers = 10  # FIXME
reso_str = str(int(np.round(delta_x/1000.))) + 'km'
outputdir = 'outputs_{:}'.format(reso_str)

t_end = 3*365*24*3600.
t_export = 1*24*3600.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 900.
    t_end = t_export
    nlayers = 4

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

coriolis_f, coriolis_beta = beta_plane_coriolis_params(37.5)
xy = SpatialCoordinate(mesh2d)
coriolis_expr = coriolis_f + coriolis_beta*(xy[1] - ly/2)
coriolis_2d = Function(P1_2d, name='coriolis').interpolate(coriolis_expr)

# temperature and salinity, results in 2.0 kg/m3 density difference
salt_const = 35.0
temp_max = 25.0
temp_ddy = -5e-6
temp_ddz = 8.2e-3

u_max = 1.5
w_max = 3e-3
u_scale = 0.5
reynolds_number = 200.
nu_scale = u_scale * delta_x / reynolds_number

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = False
options.constant_salinity = Constant(salt_const)
options.solve_temperature = True
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = True
# options.use_turbulence = True
# options.turbulence_model = 'pacanowski'
options.use_ale_moving_mesh = True
options.use_baroclinic_formulation = True
options.coriolis_frequency = coriolis_2d
options.use_lax_friedrichs_velocity = True
options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
options.use_lax_friedrichs_tracer = True
options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
options.use_limiter_for_tracers = True
options.quadratic_drag_coefficient = Constant(0.0025)
options.horizontal_viscosity_scale = Constant(nu_scale)
options.horizontal_viscosity = Constant(nu_scale)
options.horizontal_diffusivity = None  # Constant(30.)
options.vertical_viscosity = Constant(1e-2)  # tmp replacement for turbulence
# options.v_viscosity = Constant(1.0e-3)
options.vertical_diffusivity = Constant(1.0e-5)
options.output_directory = outputdir
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = Constant(u_max)
options.vertical_velocity_scale = Constant(w_max)
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.check_temperature_conservation = True
options.check_temperature_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'temp_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d', 'int_pg_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']

solver_obj.create_function_spaces()
solver_obj.create_fields()
elev_init_2d = Function(solver_obj.function_spaces.H_bhead_2d)
# nudge elevation to initial condition at the closed boundaries
nudge_swe_bnd = {'elev': elev_init_2d, 'uv': Constant((0, 0))}
bnd_id_north = 1
bnd_id_south = 2
solver_obj.bnd_functions['shallow_water'] = {
    bnd_id_north: nudge_swe_bnd,
    bnd_id_south: nudge_swe_bnd,
}

xyz = SpatialCoordinate(solver_obj.mesh)
temp_init_3d = temp_max + xyz[2]*temp_ddz + xyz[1]*temp_ddy
temp_pertubation = 0.2*sin(6*pi*xyz[0]/lx)*exp(-(xyz[1]-ly/2)**2/(ly/4)**2)

# add a relaxation term for temperature:
y_arr = Function(solver_obj.function_spaces.H).interpolate(xyz[1]).dat.data[:]
temp_relax = temp_init_3d
t_temp_relax = Constant(3.*24.*3600.)  # time scale
mask_temp_relax_3d = Function(solver_obj.function_spaces.H, name='mask_temp_relax_3d')
ly_relax = 160e3  # approx 1.5 deg
mask_numpy_y0 = (ly_relax - y_arr)/ly_relax
mask_numpy_y1 = (1 + (y_arr - ly)/ly_relax)
mask_temp_relax_3d.dat.data[:] = np.maximum(mask_numpy_y0, mask_numpy_y1)
ix = mask_temp_relax_3d.dat.data < 0
mask_temp_relax_3d.dat.data[ix] = 0.0
# File('mask.pvd').write(mask_temp_relax_3d)
# NOTE must accept ufl expressions as forcing term!
options.temperature_source_3d = mask_temp_relax_3d/t_temp_relax*(temp_relax - solver_obj.fields.temp_3d)

solver_obj.create_equations()
solver_obj.assign_initial_conditions(temp=temp_init_3d)

# compute 2D baroclinic head and use it to initialize elevation field
# to remove fast 2D gravity wave caused by the initial density difference


def compute_2d_baroc_head(solver_obj, output):
    """Computes vertical integral of baroc_head_3d"""
    compute_baroclinic_head(solver_obj)
    tmp_3d = Function(solver_obj.function_spaces.H_bhead)
    bhead_av_op = VerticalIntegrator(
        solver_obj.fields.baroc_head_3d,
        tmp_3d,
        bottom_to_top=True,
        average=True,
        elevation=solver_obj.fields.elev_cg_2d.view_3d,
        bathymetry=solver_obj.fields.bathymetry_2d.view_3d)
    bhead_surf_extract_op = SubFunctionExtractor(
        tmp_3d,
        output,
        boundary='top', elem_facet='top')
    bhead_av_op.solve()
    bhead_surf_extract_op.solve()


compute_2d_baroc_head(solver_obj, elev_init_2d)
elev_init_2d *= -1  # flip sign => total pressure gradient is zero
mean_elev = assemble(elev_init_2d*dx)/lx/ly
elev_init_2d += -mean_elev  # remove mean

# initialize flow in geostrophic balance
solver_obj.assign_initial_conditions(temp=temp_init_3d + temp_pertubation, elev=elev_init_2d)

print_output('Resolution: {:}'.format(delta_x))
print_output('Reynolds number: {:}'.format(reynolds_number))
print_output('Horizontal viscosity: {:}'.format(nu_scale))
print_output('Exporting to {:}'.format(outputdir))

solver_obj.iterate()
