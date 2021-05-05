"""
Meander Test case
=======================
Solves the test case of flow around a 180 degree bend replicating lab experiment 4 in Yen & Lee (1995).
We use this test case to validate the implementation of the mathematical and
numerical methods used in Thetis to model sediment transport and morphological changes.
Specifically this test case tests the secondary current implementation.

For more details, see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *
# import bathymetry and mesh for meander
from meander_setup import *
# Note it is necessary to run meander_hydro first to get the hydrodynamics simulation


def update_forcings_bnd(t_new):

    if t_new != t_old.dat.data[:]:
        # update boundary condtions
        if t_new*morfac <= 6000:
            elev_constant.assign(gradient_elev*t_new*morfac + elev_init_const)
            flux_constant.assign((gradient_flux*t_new*morfac) - 0.02)
        else:
            flux_constant.assign((gradient_flux2*(t_new*morfac-6000)) - 0.053)
            elev_constant.assign(gradient_elev2*(t_new*morfac-18000) + elev_init_const)
        t_old.assign(t_new)


t_old = Constant(0.0)

# define function spaces
DG_2d = FunctionSpace(mesh2d, 'DG', 1)
vector_dg = VectorFunctionSpace(mesh2d, 'DG', 1)

# choose directory to output results
outputdir = 'outputs'

morfac = 50
dt = 2
end_time = 5*3600
viscosity_hydro = Constant(5*10**(-2))

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # the example is being run as a test
    # run the spin-up by importing it
    import meander_hydro  # NOQA
    end_time = 1800.

# initialise velocity and elevation
with DumbCheckpoint("hydrodynamics_meander/elevation", mode=FILE_READ) as chk:
    elev = Function(DG_2d, name="elevation")
    chk.load(elev)
    chk.close()
with DumbCheckpoint('hydrodynamics_meander/velocity', mode=FILE_READ) as chk:
    uv = Function(vector_dg, name="velocity")
    chk.load(uv)
    chk.close()

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options

# this test case only uses bedload transport but using all slope effect corrections and secondary current
options.sediment_model_options.solve_suspended_sediment = False
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True
options.sediment_model_options.use_angle_correction = True
options.sediment_model_options.use_slope_mag_correction = True
options.sediment_model_options.use_secondary_current = True
options.sediment_model_options.use_advective_velocity_correction = False
options.sediment_model_options.morphological_viscosity = Constant(1e-6)
options.sediment_model_options.average_sediment_size = Constant(10**(-3))
options.sediment_model_options.bed_reference_height = Constant(0.003)
options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

options.simulation_end_time = end_time/morfac
options.simulation_export_time = options.simulation_end_time/45
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']
# using nikuradse friction
options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)
# set horizontal viscosity parameter
options.horizontal_viscosity = Constant(viscosity_hydro)
# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

left_bnd_id = 1
right_bnd_id = 2

# set boundary conditions
gradient_flux = (-0.053 + 0.02)/6000
gradient_flux2 = (-0.02+0.053)/(18000-6000)
gradient_elev = (10.04414-9.9955)/6000
gradient_elev2 = (9.9955-10.04414)/(18000-6000)
elev_init_const = (-max(bathymetry_2d.dat.data[:]) + 0.05436)
flux_constant = Constant(-0.02)
elev_constant = Constant(elev_init_const)

swe_bnd = {}
swe_bnd[3] = {'un': Constant(0.0)}
swe_bnd[left_bnd_id] = {'flux': flux_constant}
swe_bnd[right_bnd_id] = {'elev': elev_constant}
solver_obj.bnd_functions['shallow_water'] = swe_bnd
# set initial conditions
solver_obj.assign_initial_conditions(uv=uv, elev=elev)

# run model
solver_obj.iterate(update_forcings=update_forcings_bnd)
