""""
We consider a simple tidal lagoon - no sluice gate boundaries are included so we only focus on the control of turbines.
This example presents the tidal power plant simulation functionality by coupling operational algorithms with Thetis
boundaries

More information about the tidal range algorithms can be found in:

Angeloudis A, Kramer SC, Avdis A & Piggott MD, Optimising tidal range power plant operation,
Applied Energy,212,680-690, 212, https://doi.org/10.1016/j.apenergy.2017.12.052
"""
from thetis import *

# modules.tools contains the LagoonCallback that is called during the simulation
from modules.tools import *

# input_barrages contains functions to initialise the lagoon status and generic control parameters for the operation
from modules import input_barrages

mesh2d = Mesh('lagoon.msh')

# Initial functions for lagoon
lagoon_input = input_barrages.input_predefined_barrage_specs(turbine_number=25, sluice_number=0, operation='two-way')

t_end = 48 * 3600.  # total duration in seconds
t_export = 200.0    # export interval in seconds
Dt = 25             # timestep

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = t_export

# Omega for amplitude of seaward boundary - assumed to be a sinusoidal function with a given amplitude
amplitude = 4.0
period = 12.42 * 3600
omega = (2 * pi) / period

# Bathymetry and viscosity field
P1_2d = get_functionspace(mesh2d, 'DG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
viscosity_2d = Function(P1_2d, name='viscosity')

# Bathymetry combination of bell curve and sloping bathymetry
x, y = SpatialCoordinate(mesh2d)

depth_oce = 20.0
depth_riv = -10.0
mi, sigma = 0, 2000
bathymetry_2d.interpolate(-2 * 1e5 * (-1 / (sigma * sqrt(2 * pi)) * exp(-(y - 3000 - mi)**2 / (2 * sigma**2)))
                          + (depth_riv - depth_oce) * x/16e3)

# Viscosity sponge at the seaward boundary
viscosity_2d.interpolate(conditional(le(x, 2e3), 1e3 * (2e3+1 - x)/2e3, 1))

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.check_volume_conservation_2d = True
options.timestepper_type = 'CrankNicolson'
options.fields_to_export = ['uv_2d', 'elev_2d']
options.timestepper_options.implicitness_theta = 0.5
options.timestepper_options.use_semi_implicit_linearization = True
options.use_wetting_and_drying = True
options.wetting_and_drying_alpha = Constant(0.5)
options.manning_drag_coefficient = Constant(0.02)
options.horizontal_viscosity = viscosity_2d
options.timestep = Dt

# set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_init.assign(0.0)

tidal_elev = Function(bathymetry_2d.function_space())

# Initialise hydraulic structure boundaries  and max them as fluxes
lagoon_hydraulic_structures = {"sl_i": Constant(0.), "sl_o": Constant(0.), "tb_i": Constant(0.), "tb_o": Constant(0.)}

solver_obj.bnd_functions['shallow_water'] = {5: {'elev': tidal_elev},
                                             2: {'flux': lagoon_hydraulic_structures["tb_i"]},
                                             1: {'flux': lagoon_hydraulic_structures["tb_o"]}, }

solver_obj.assign_initial_conditions(uv=as_vector((1e-5, 0.0)), elev=elev_init)

# Create tidal lagoon callback
# dx(1) is outer location, dx(2) is inner location (marked areas using gmsh)
cb_lagoon = LagoonCallback(solver_obj, {"inner": dx(2), "outer": dx(1)}, lagoon_input,
                           thetis_boundaries=lagoon_hydraulic_structures,
                           number_timesteps=5, name="lagoon_1", export_to_hdf5=True)

solver_obj.add_callback(cb_lagoon, 'timestep')


def update_forcings(t_new,):
    tidal_elev.assign(Constant(tanh((t_new)/(4*3600.)) * amplitude * sin(omega * t_new)))


# Solve the system
solver_obj.iterate(update_forcings=update_forcings)
