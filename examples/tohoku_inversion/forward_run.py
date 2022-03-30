from thetis import *
import time as time_mod
from model_config import *
import argparse
import os


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami propagation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-s", "--source-model", type=str, default="CG1")
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()
source_model = args.source_model
suffix = args.suffix
no_exports = os.getenv("THETIS_REGRESSION_TEST") is not None

# Setup initial condition
pwd = os.path.abspath(os.path.dirname(__file__))
mesh2d = Mesh(f"{pwd}/japan_sea.msh")
elev_init, controls = initial_condition(mesh2d, source_model=source_model)

# Solve forward
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir = f"{pwd}/outputs_forward_{source_model}"
if suffix != "":
    output_dir = "_".join([output_dir, suffix])
solver_obj = construct_solver(
    elev_init,
    output_directory=output_dir,
    store_station_time_series=not no_exports,
    no_exports=no_exports,
)
print_output(f"Exporting to {solver_obj.options.output_directory}")
tic = time_mod.perf_counter()
solver_obj.iterate()
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
