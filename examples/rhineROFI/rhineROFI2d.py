"""
Rhine ROFI test case
====================

Idealized Rhine river plume test case according to [2].

Salinity ranges from 32 psu in ocean to 0 in river.
Temperature is constant 10 deg Celcius. This corresponds to density
values 1024.611 kg/m3 in the ocean and 999.702 in the river.

[1] de Boer, G., Pietrzak, J., and Winterwerp, J. (2006). On the vertical
    structure of the Rhine region of freshwater influence. Ocean Dynamics,
    56(3):198-216.
[2] Fischer, E., Burchard, H., and Hetland, R. (2009). Numerical
    investigations of the turbulent kinetic energy dissipation rate in the
    Rhine region of freshwater influence. Ocean Dynamics, 59:629-641.
"""
from thetis import *

reso = 'coarse'
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    reso = 'test'

outputdir = 'outputs_2d_{:}'.format(reso)
mesh2d = Mesh('mesh_rhineRofi_{:}.msh'.format(reso))
print_output('Loaded mesh '+mesh2d.name)
print_output('Exporting to '+outputdir)

# Physical parameters
eta_amplitude = 1.00  # mean (Fisher et al. 2009 tidal range 2.00 )
eta_phase = 0
H = 20  # water depth
HInlet = 5  # water depth at river inlet
Lriver = 45e3
Wriver = 500
Qriver = 3.0e3  # 1.5e3 river discharge (Fisher et al. 2009)
Sriver = 0
Ssea = 32
density_river = 999.7
density_ocean = 1024.6
Ttide = 44714.0  # M2 tidal period (Fisher et al. 2009)
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaEarth = 2*np.pi/Tday
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H)  # [m/s] wave speed
lat_deg = 52.5  # latitude
phi = (np.pi/180)*lat_deg  # latitude in radians
coriolis_f = 2*OmegaEarth*sin(phi)  # [rad/s] Coriolis parameter ~ 1.1e-4
kelvin_k = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvin_m = (coriolis_f/c)  # [-] Cross-shore variation

dt = 8.0
t_end = 32*44714
t_export = 900.0  # 44714/12

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 10 * dt
    t_end = t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
x, y = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(conditional(x > 0.0,
                                      H*(1-x/Lriver) + HInlet*(x/Lriver),
                                      H))

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.coriolis_frequency = Constant(coriolis_f)
options.horizontal_viscosity = Constant(10.0)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.timestepper_type = 'CrankNicolson'
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(1.5)
options.fields_to_export = ['uv_2d', 'elev_2d']

bnd_elev = Function(P1_2d, name='Boundary elevation')
bnd_time = Constant(0)
xyz = solver_obj.mesh2d.coordinates
tri = TrialFunction(P1_2d)
test = TestFunction(P1_2d)
elev = eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*elev*dx
bnd_elev_prob = LinearVariationalProblem(a, L, bnd_elev)
bnd_elev_solver = LinearVariationalSolver(bnd_elev_prob)
bnd_elev_solver.solve()

# fs = P1_2d
# bnd_v = Function(fs, name='Boundary v velocity')
# tri = TrialFunction(fs)
# test = TestFunction(fs)
# v = -(g*kelvin_k/OmegaTide)*eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
# a = inner(test, tri)*dx
# L = test*v*dx
# bnd_v_prob = LinearVariationalProblem(a, L, bnd_v)
# bnd_v_solver = LinearVariationalSolver(bnd_v_prob)
# bnd_v_solver.solve()

river_discharge = Constant(-Qriver)
tide_elev_funcs = {'elev': bnd_elev}
# tide_uv_funcs = {'un': bnd_v}
# tide_funcs = {'elev': bnd_elev, 'un': bnd_v}
open_funcs = {'radiation': None}
river_funcs = {'flux': river_discharge}
solver_obj.bnd_functions['shallow_water'] = {1: tide_elev_funcs,
                                             2: tide_elev_funcs,
                                             3: tide_elev_funcs,
                                             6: river_funcs}

# TODO set correct boundary conditions
solver_obj.create_equations()
elev_init = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init.interpolate(conditional(le(x, 0.0),
                                  eta_amplitude*exp((x)*kelvin_m)*cos(y*kelvin_k),
                                  eta_amplitude*cos(y*kelvin_k)))
elev_init2 = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init2.interpolate(conditional(le(x, 0.0),
                                   eta_amplitude*exp(x*kelvin_m)*cos(y*kelvin_k),
                                   0.0))
uv_init = Function(solver_obj.function_spaces.U_2d, name='initial velocity')
# uv_init.interpolate(conditional(le(x, 0.0),
#                                 eta_amplitude*exp(x*kelvin_m)*cos(y*kelvin_k),
#                                 eta_amplitude*cos(y*kelvin_k)))
tri = TrialFunction(solver_obj.function_spaces.U_2d)
test = TestFunction(solver_obj.function_spaces.U_2d)
a = inner(test, tri)*dx
uv = (g*kelvin_k/OmegaTide)*elev_init2
l = test[1]*uv*dx
solve(a == l, uv_init)


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_solver.solve()
    # bnd_v_solver.solve()


solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)
solver_obj.iterate(update_forcings=update_forcings)
