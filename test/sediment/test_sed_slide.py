"""
Unphysical Slope Test case
=======================

Tests the sediment slide mechanism in the Exner equation, by starting with an
unphysical slope and reducing the slope angle over time

"""

from thetis import *

# define mesh
mesh2d = RectangleMesh(20, 10, 4, 2)
x, y = SpatialCoordinate(mesh2d)

vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
V = FunctionSpace(mesh2d, 'CG', 1)

# define initial bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
z_init = conditional(x < 2, 0, conditional(x <= 4, 0.5*x-1, 0))
bathymetry_2d.interpolate(z_init)

# define initial conditions
uv_init = Function(vectorP1_2d).interpolate(as_vector((Constant(0.46), Constant(0.0))))
elev_init = Constant(4)

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = 1
options.simulation_end_time = 20
options.no_exports = True

options.horizontal_viscosity = Constant(1e-6)

# for the test, only using bedload with sediment slide mechanism
options.sediment_model_options.solve_suspended_sediment = False
options.sediment_model_options.use_bedload = True
options.sediment_model_options.use_slope_mag_correction = False
options.sediment_model_options.use_angle_correction = False
options.sediment_model_options.use_sediment_slide = True
options.sediment_model_options.solve_exner = True
options.sediment_model_options.average_sediment_size = Constant(2.6e-4)
options.sediment_model_options.bed_reference_height = Constant(0.0002)
# average meshgrid stepsize
options.sediment_model_options.meshgrid_size = Constant(0.2)
# maximum angle of repose which the slope should have (this is the target angle)
options.sediment_model_options.max_angle = Constant(22)
options.sediment_model_options.morphological_acceleration_factor = Constant(20)
options.sediment_model_options.use_advective_velocity_correction = False
# using nikuradse friction
options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.timestep = 0.1

# set boundary conditions
left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

uv_vector = as_vector((0.46, 0.0))

swe_bnd[left_bnd_id] = {'uv': uv_vector}

swe_bnd[right_bnd_id] = {'elev': Constant(4)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

# set initial conditions
solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

beta = Function(V)
max_beta_list = []


def update_forcing(t_new):
    # record maximum slope angle and check it is decreasing
    beta.interpolate(solver_obj.sediment_model.betaangle)
    max_beta_list.append(max(beta.dat.data[:])*180/pi)

    if len(max_beta_list) > 30:
        assert max_beta_list[-1] < max_beta_list[-10], 'Sediment slide mechanism is not causing\
                                                         the angle to decrease'


solver_obj.iterate(update_forcings=update_forcing)

assert np.round(max_beta_list[-1], 1) == 24.6, 'Sediment slide mechanism has changed'
