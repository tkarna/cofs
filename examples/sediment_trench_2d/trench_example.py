"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench.

We use this test case to validate the implementation of the mathematical and
numerical methods used in Thetis to model sediment transport and morphological changes.
In the figure produced, we compare our results with experimental data from a lab study

For more details, see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *

import numpy as np
import matplotlib.pyplot as plt

conservative = False

# Note it is necessary to run trench_hydro first to get the hydrodynamics simulation

# define mesh
lx = 16
ly = 1.1
nx = lx*5
ny = 5
mesh2d = RectangleMesh(nx, ny, lx, ly)

x, y = SpatialCoordinate(mesh2d)

# define function spaces
V = FunctionSpace(mesh2d, "CG", 1)
DG_2d = FunctionSpace(mesh2d, "DG", 1)
vector_dg = VectorFunctionSpace(mesh2d, "DG", 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='bathymetry_2d')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                     conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
bathymetry_2d.interpolate(-trench)

# choose directory to output results
outputdir = 'outputs'

print_output('Exporting to '+outputdir)

morfac = 100
dt = 0.3
end_time = 15*3600

diffusivity = 0.15
viscosity_hydro = Constant(1e-6)

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # the example is being run as a test
    # run the spin-up by importing it
    import trench_hydro  # NOQA
    end_time = 3600.

# initialise velocity and elevation
chk = DumbCheckpoint("hydrodynamics_trench/elevation", mode=FILE_READ)
elev = Function(DG_2d, name="elevation")
chk.load(elev)
chk.close()

chk = DumbCheckpoint('hydrodynamics_trench/velocity', mode=FILE_READ)
uv = Function(vector_dg, name="velocity")
chk.load(uv)
chk.close()

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options

options.sediment_model_options.solve_suspended_sediment = True
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True

options.sediment_model_options.use_sediment_conservative_form = conservative
options.sediment_model_options.average_sediment_size = Constant(160*(10**(-6)))
options.sediment_model_options.bed_reference_height = Constant(0.025)
options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

options.simulation_end_time = end_time/morfac
options.simulation_export_time = options.simulation_end_time/45

options.output_directory = outputdir
options.check_volume_conservation_2d = True

if options.sediment_model_options.solve_suspended_sediment:
    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d', 'bathymetry_2d']  # note exporting bathymetry must be done through export func
    options.sediment_model_options.check_sediment_conservation = True
else:
    options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']  # note exporting bathymetry must be done through export func

# using nikuradse friction
options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)

# set horizontal diffusivity parameter
options.horizontal_diffusivity = Constant(diffusivity)
options.horizontal_viscosity = Constant(viscosity_hydro)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.norm_smoother = Constant(0.1)

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

if options.sediment_model_options.solve_suspended_sediment:
    # setting an equilibrium boundary conditions results in the sediment value at the boundary
    # being chosen so that erosion and deposition are equal here (ie. in equilibrium) and the bed is immobile at this boundary
    solver_obj.bnd_functions['sediment'] = {
        left_bnd_id: {'flux': Constant(-0.22), 'equilibrium': None},
        right_bnd_id: {'elev': Constant(0.397)}}

    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)

else:
    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)

# run model
solver_obj.iterate()

# record final bathymetry for plotting
xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 15.8, 80):
    xaxisthetis1.append(i)
    if conservative:
        baththetis1.append(-solver_obj.fields.bathymetry_2d.at([i, 0.55]))
    else:
        baththetis1.append(-solver_obj.fields.bathymetry_2d.at([i, 0.55]))

if os.getenv('THETIS_REGRESSION_TEST') is None:
    # Compare model and experimental results
    # (this part is skipped when run as a test)
    data = np.genfromtxt('experimental_data.csv', delimiter=',')

    plt.scatter([i[0] for i in data], [i[1] for i in data], label='Experimental Data')

    plt.plot(xaxisthetis1, baththetis1, label='Thetis')
    plt.legend()
    plt.show()
