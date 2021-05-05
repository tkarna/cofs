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


class FreshwaterConservationCallback(DiagnosticCallback):
    """Checks conservation of freshwater"""
    name = 'fresh water volume'
    variable_names = ['integral', 'difference']

    def __init__(self, ref_salinity, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        Creates fresh water conservation check callback object

        ref_salinity : float
            constant maximum salinity value used to compute freshwater tracer
        """
        super(FreshwaterConservationCallback, self).__init__(solver_obj,
                                                             outputdir=outputdir,
                                                             export_to_hdf5=export_to_hdf5,
                                                             append_to_log=append_to_log)
        self.ref_salinity = ref_salinity

        def mass():
            freshwater = 1.0 - self.solver_obj.fields['salt_3d']/self.ref_salinity
            return comp_tracer_mass_3d(freshwater)

        self.scalar_callback = mass
        self.previous_value = None

    def __call__(self):
        value = self.scalar_callback()
        if self.previous_value is None:
            self.previous_value = value
        diff = (value - self.previous_value)
        self.previous_value = value
        return value, diff

    def message_str(self, *args):
        line = '{0:s} {1:11.4e}, diff {2:11.4e}'.format(self.name, args[0], args[1])
        return line


# set physical constants
physical_constants['rho0'].assign(1000.0)

reso = 'coarse'
layers = 12
if reso == 'fine':
    layers = 30  # NOTE 40 in [2]

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    reso = 'test'
    layers = 2

outputdir = 'outputs_{:}'.format(reso)
mesh2d = Mesh('mesh_rhineRofi_{:}.msh'.format(reso))
print_output('Loaded mesh ' + mesh2d.name)
print_output('Exporting to ' + outputdir)

# Physical parameters
eta_amplitude = 1.00  # tidal range 2.00; mean scenario in [2]
eta_phase = 0
H_ocean = 20  # water depth
H_river = 5  # water depth at river inlet
L_river = 45e3  # NOTE L_river is 75 km in [2]
Q_river = 1.5e3  # 1.5e3 or 2.2e3 river discharge [2]
temp_const = 10.0
salt_river = 0.0
salt_ocean = 32.0

Ttide = 44714.0  # M2 tidal period [2]
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaEarth = 2*np.pi/Tday
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H_ocean)  # [m/s] wave speed
lat_deg = 52.5  # latitude
phi = (np.pi/180)*lat_deg  # latitude in radians
coriolis_f = 2*OmegaEarth*sin(phi)  # [rad/s] Coriolis parameter ~ 1.1e-4
kelvin_k = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvin_m = (coriolis_f/c)  # [-] Cross-shore variation

dt = 7.0
t_end = 34*Ttide
t_export = Ttide/40  # approx 18.6 min

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_export = 10 * dt
    t_end = t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
x, y = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(conditional(le(x, 0.0),
                                      H_ocean,
                                      H_ocean*(1-x/L_river) + H_river*(x/L_river)))

simple_barotropic = False  # for debugging

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'LeapFrog'
options.solve_salinity = not simple_barotropic
options.solve_temperature = False
options.constant_temperature = Constant(temp_const)
options.use_implicit_vertical_diffusion = not simple_barotropic
options.use_bottom_friction = not simple_barotropic
options.bottom_roughness = Constant(0.005)
options.use_turbulence = not simple_barotropic
options.use_turbulence_advection = not simple_barotropic
# options.use_ale_moving_mesh = False
options.use_baroclinic_formulation = not simple_barotropic
options.use_lax_friedrichs_velocity = True
options.use_lax_friedrichs_tracer = True
# options.horizontal_diffusivity = Constant(50.0)
# options.horizontal_viscosity = Constant(50.0)
options.vertical_viscosity = Constant(1.3e-6)  # background value
options.vertical_diffusivity = Constant(1.4e-7)  # background value
options.use_limiter_for_tracers = True
Re_h = 5.0
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(Re_h))
options.coriolis_frequency = Constant(coriolis_f)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(2.0)
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'salt_3d', 'smag_visc_3d',
                                 'eddy_visc_3d', 'shear_freq_3d',
                                 'buoy_freq_3d', 'tke_3d', 'psi_3d',
                                 'eps_3d', 'len_3d']

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

fs = P1_2d
bnd_v = Function(fs, name='Boundary v velocity')
tri = TrialFunction(fs)
test = TestFunction(fs)
v = -(g*kelvin_k/OmegaTide)*eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*v*dx
bnd_v_prob = LinearVariationalProblem(a, L, bnd_v)
bnd_v_solver = LinearVariationalSolver(bnd_v_prob)
bnd_v_solver.solve()

river_discharge = Constant(-Q_river)
ocean_salt = Constant(salt_ocean)
river_salt = Constant(salt_river)
tide_elev_funcs = {'elev': bnd_elev}
tide_uv_funcs = {'un': bnd_v}
open_funcs = {'symm': None}
river_funcs = {'flux': river_discharge}
bnd_ocean_salt = {'value': ocean_salt}
bnd_river_salt = {'value': river_salt}
solver_obj.bnd_functions['shallow_water'] = {1: tide_elev_funcs, 2: tide_elev_funcs,
                                             3: tide_elev_funcs, 6: river_funcs}
solver_obj.bnd_functions['momentum'] = {1: open_funcs, 2: open_funcs,
                                        3: open_funcs, 6: open_funcs}
solver_obj.bnd_functions['salt'] = {1: bnd_ocean_salt, 2: bnd_ocean_salt,
                                    3: bnd_ocean_salt, 6: bnd_river_salt}

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
tri = TrialFunction(solver_obj.function_spaces.U_2d)
test = TestFunction(solver_obj.function_spaces.U_2d)
a = inner(test, tri)*dx
uv = (g*kelvin_k/OmegaTide)*elev_init2
l = test[1]*uv*dx
solve(a == l, uv_init)
salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
xoff = 10.5e3
sigma = 2000.0
x, y, z = SpatialCoordinate(solver_obj.mesh)
salt_init3d.interpolate(salt_ocean - (salt_ocean - salt_river)*(1 + tanh((x - xoff)/sigma))/2)


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_solver.solve()


solver_obj.add_callback(FreshwaterConservationCallback(salt_ocean,
                                                       solver_obj,
                                                       export_to_hdf5=True,
                                                       append_to_log=True))

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, uv_2d=uv_init)
solver_obj.iterate(update_forcings=update_forcings)
