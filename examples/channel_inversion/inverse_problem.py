from thetis import *
import firedrake as fd
from firedrake.adjoint import *
import numpy
import thetis.inversion_tools as inversion_tools
from model_config import construct_solver
import argparse
import os

# parse user input
parser = argparse.ArgumentParser(
    description='Channel inversion problem',
    # includes default values in help entries
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--controls', nargs='+',
                    help='Control variable(s) to optimize',
                    choices=['Bathymetry', 'Manning', 'InitialElev'],
                    default=['Bathymetry'],
                    )
parser.add_argument('--no-consistency-test', action='store_true',
                    help='Skip consistency test')
parser.add_argument('--no-taylor-test', action='store_true',
                    help='Skip Taylor test')
args = parser.parse_args()
controls = sorted(args.controls)
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test
no_exports = os.getenv('THETIS_REGRESSION_TEST') is not None

# Set output directory
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir = f'{pwd}/outputs_new_' + '-'.join(controls) + '-opt'

# annotate all Firedrake operations of the forward run
continue_annotation()

# Create the solver object
# NOTE We disable all exports here as they would only be active during
# the initial forward run
solver_obj = construct_solver(
    output_directory=output_dir,
    store_station_time_series=False,
    no_exports=True,
)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d
manning_2d = solver_obj.fields.manning_2d
elev_init_2d = solver_obj.fields.elev_init_2d


# Setup controls and regularization parameters
gamma_hessian_list = []
control_bounds_list = []
for control_name in controls:
    if control_name == 'Bathymetry':
        bathymetry_2d.assign(5.0)
        bounds = [1.0, 50.]
        gamma_hessian = Constant(1e-3)  # regularization parameter
    elif control_name == 'Manning':
        manning_2d.assign(1.0e-03)
        bounds = [1e-4, 1e-1]
        gamma_hessian = Constant(1.0)
    elif control_name == 'InitialElev':
        elev_init_2d.assign(0.5)
        bounds = [-10., 10.]
        gamma_hessian = Constant(0.1)
    else:
        raise ValueError(f'Unsupported control variable {control_name}')
    print_output(f'{control_name} regularization params: hess={float(gamma_hessian):.3g}')
    gamma_hessian_list.append(gamma_hessian)
    control_bounds_list.append(bounds)
# reshape to [[lo1, lo2, ...], [hi1, hi2, ...]]
control_bounds = numpy.array(control_bounds_list).T

# Assign initial conditions
print_output('Exporting to ' + options.output_directory)
solver_obj.assign_initial_conditions(elev=elev_init_2d, uv=Constant((1e-5, 0)))

# Create station manager
observation_data_dir = f'{pwd}/outputs_forward'
variable = 'elev'
station_names = [
    'stationA',
    'stationB',
    'stationC',
    'stationD',
    'stationE',
]
sta_manager = inversion_tools.StationObservationManager(
    mesh2d, output_directory=options.output_directory)
# TODO: Update scaling to depend on number of DOFs in the problem
cost_function_scaling = domain_constant(100000 * solver_obj.dt / options.simulation_end_time, mesh2d)
sta_manager.cost_function_scaling = cost_function_scaling
sta_manager.load_observation_data(observation_data_dir, station_names, variable)
sta_manager.set_model_field(solver_obj.fields.elev_2d)

# Define the scaling for the cost function so that dJ/dm ~ O(1)
J_scalar = sta_manager.cost_function_scaling

# Create inversion manager and add controls
inv_manager = inversion_tools.InversionManager(
    sta_manager, output_dir=options.output_directory, no_exports=no_exports,
    penalty_parameters=gamma_hessian_list, cost_function_scaling=J_scalar,
    test_consistency=do_consistency_test, test_gradient=do_taylor_test)
for control_name in controls:
    if control_name == 'Bathymetry':
        inv_manager.add_control(bathymetry_2d)
    elif control_name == 'Manning':
        inv_manager.add_control(manning_2d)
    elif control_name == 'InitialElev':
        inv_manager.add_control(elev_init_2d)

# Extract the regularized cost function
cost_function = inv_manager.get_cost_function(solver_obj)

# Solve and setup reduced functional
solver_obj.iterate(export_func=cost_function)  # note that export time should equal dt if not using a custom callback
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1  # scipy diagnostics -1, 0, 1, 99, 100, 101
opt_options = {
    'maxiter': 6,  # NOTE increase to run iteration longer
    'ftol': 1e-5,
    'disp': opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    opt_options['maxiter'] = 1
control_opt_list = inv_manager.minimize(
    opt_method='L-BFGS-B', bounds=control_bounds, **opt_options)
if isinstance(control_opt_list, Function):
    control_opt_list = [control_opt_list]
for oc, cc in zip(control_opt_list, inv_manager.control_coeff_list):
    name = cc.name()
    oc.rename(name)
    print_function_value_range(oc, prefix='Optimal')
    if not no_exports:
        fd.VTKFile(f'{options.output_directory}/{name}_optimised.pvd').write(oc)
