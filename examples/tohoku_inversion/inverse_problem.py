from thetis import *
from firedrake.adjoint import *
import thetis.inversion_tools as inversion_tools
from model_config import *
import argparse
import os
import sys


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami source inversion problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-s", "--source-model", type=str, default="okada")
parser.add_argument(
    "-o",
    "--okada-parameters",
    help="Okada parameters to invert for in the Okada model case",
    nargs="+",
    choices=["depth", "dip", "slip", "rake"],
    default=["depth", "dip", "slip", "rake"],
)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--ftol", type=float, default=1.0e-05)
parser.add_argument("--no-regularization", action="store_true")
parser.add_argument("--no-consistency-test", action="store_true")
parser.add_argument("--no-taylor-test", action="store_true")
parser.add_argument("--suffix", type=str, default=None)
args = parser.parse_args()
source_model = args.source_model
if source_model == "okada" and len(args.okada_parameters) == 0:
    print_output("Specify control parameters using the --okada-parameters option.")
    sys.exit(0)
no_regularization = args.no_regularization
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test
suffix = args.suffix

# annotate all Firedrake operations of the forward run
continue_annotation()

# Setup initial condition
pwd = os.path.abspath(os.path.dirname(__file__))
with CheckpointFile(f"{pwd}/japan_sea_bathymetry.h5", "r") as f:
    mesh2d = f.load_mesh("firedrake_default")
    bathymetry = f.load_function(mesh2d, "Bathymetry")
source = get_source(mesh2d, source_model)
if source_model == "okada":
    source.subfault_variables = args.okada_parameters

# Setup PDE
output_dir = f"{pwd}/outputs_elev-init-optimization_{source_model}"
if suffix is not None:
    output_dir = "_".join([output_dir, suffix])
solver_obj = construct_solver(
    bathymetry, source.elev_init,
    output_directory=output_dir,
    store_station_time_series=False,
    no_exports=True,
)
options = solver_obj.options
if not options.no_exports:
    print_output(f"Exporting to {options.output_directory}")

# Set up observation and regularization managers
observation_data_dir = f"{pwd}/observations"
variable = "elev"
stations = read_station_data()
station_names = list(stations.keys())
start_times = [dat["start"] for dat in stations.values()]
end_times = [dat["end"] for dat in stations.values()]
sta_manager = inversion_tools.StationObservationManager(
    mesh2d, output_directory=options.output_directory
)
sta_manager.load_elev_observation_data(
    observation_data_dir,
    station_names,
    variable,
    start_times=start_times,
    end_times=end_times,
)
sta_manager.set_model_field(solver_obj.fields.elev_2d)

# Define the scaling for the cost function so that J ~ O(1)
J_scalar = Constant(solver_obj.dt / options.simulation_end_time)

# Create inversion manager and add controls
no_exports = os.getenv("THETIS_REGRESSION_TEST") is not None
real_mode = source_model in ("box", "radial", "okada")
gamma = 0 if no_regularization else 1e-04 if real_mode else 1e-01
inv_manager = inversion_tools.InversionManager(
    sta_manager, real=real_mode, cost_function_scaling=J_scalar,
    penalty_parameters=[Constant(gamma) for c in source.controls],
    output_dir=options.output_directory, no_exports=no_exports,
    test_consistency=do_consistency_test, test_gradient=do_taylor_test)
for c in source.controls:
    inv_manager.add_control(c)

# Extract the regularized cost function
cost_function = inv_manager.get_cost_function(solver_obj, weight_by_variance=True)
cost_function_callback = inversion_tools.CostFunctionCallback(solver_obj, cost_function)
solver_obj.add_callback(cost_function_callback, 'timestep')

# Solve and setup the reduced functional
solver_obj.iterate()
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1
opt_options = {
    "maxiter": args.maxiter,
    "ftol": args.ftol,
    "disp": opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv("THETIS_REGRESSION_TEST") is not None:
    opt_options["maxiter"] = 1
control_opt_list = inv_manager.minimize(
    opt_method="L-BFGS-B", bounds=source.control_bounds, **opt_options)
if options.no_exports:
    sys.exit(0)
source = get_source(mesh2d, source_model, initial_guess=inv_manager.control_coeff_list)
if source_model == "okada":
    source.subfault_variables = args.okada_parameters
outfile = VTKFile(f"{options.output_directory}/elevation_optimised.pvd")
print_function_value_range(source.elev_init, prefix="Optimal")
outfile.write(source.elev_init)
