"""
Meander Test case
=======================
Solves the initial hydrodynamics simulation of flow around a 180 degree bend replicating
lab experiment 4 in Yen & Lee (1995).
Note this is not the main run-file and is just used to create an initial checkpoint for
the morphodynamic simulation.

For more details of the test case set-up see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *
import numpy as np
# import bathymetry and mesh for meander
from meander_setup import *

# define function spaces
P1_2d = FunctionSpace(mesh2d, 'DG', 1)
vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)

# simulate initial hydrodynamics
# define initial elevation
elev_init = Function(P1_2d).interpolate(0.0544 - bathymetry_2d)
# define initial velocity
uv_init = Function(vectorP1_2d).interpolate(as_vector((0.001, 0.001)))

# choose directory to output results
outputdir = 'outputs'
print_output('Exporting to '+outputdir)
t_end = 200
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # run as tests, not sufficient for proper spin up
    # but we simply want a run-through-without-error test
    t_end = 25
# export interval in seconds
t_export = np.round(t_end/40, 0)
# define parameters
average_size = 10**(-3)
ksp = Constant(3*average_size)

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.solve_tracer = False
options.use_lax_friedrichs_tracer = False
# using nikuradse friction
options.nikuradse_bed_roughness = ksp
# setting viscosity
options.horizontal_viscosity = Constant(5*10**(-2))
# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 1

# set boundary conditions
elev_init_const = (-max(bathymetry_2d.dat.data[:]) + 0.05436)
left_bnd_id = 1
right_bnd_id = 2
swe_bnd = {}
swe_bnd[3] = {'un': Constant(0.0)}
swe_bnd[1] = {'flux': Constant(-0.02)}
swe_bnd[2] = {'elev': Constant(elev_init_const), 'flux': Constant(0.02)}
solver_obj.bnd_functions['shallow_water'] = swe_bnd
solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

# run model
solver_obj.iterate()

# store hydrodynamics for next simulation
uv, elev = solver_obj.fields.solution_2d.split()
checkpoint_dir = "hydrodynamics_meander"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with DumbCheckpoint(checkpoint_dir + '/velocity', mode=FILE_CREATE) as chk:
    chk.store(uv, name="velocity")
    chk.close()
with DumbCheckpoint(checkpoint_dir + "/elevation", mode=FILE_CREATE) as chk:
    chk.store(elev, name="elevation")
    chk.close()
