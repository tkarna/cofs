"""
Runs all example scripts. Only tests whether examples can be executed.
"""
import pytest
import os
import subprocess
import glob
import sys
import shutil

# set environment flag
# can be used in examples to reduce cpu cost
os.environ['THETIS_REGRESSION_TEST'] = "1"

exclude_files = [
    'baroclinic_eddies/diagnostics.py',
    'baroclinic_eddies/submitRuns.py',
    'bottomFriction/plot_results.py',
    'columbia_plume/atm_forcing.py',
    'columbia_plume/bathymetry.py',
    'columbia_plume/cre-plume.py',
    'columbia_plume/diagnostics.py',
    'columbia_plume/ncom_forcing.py',
    'columbia_plume/plot_elevation_ts.py',
    'columbia_plume/plot_salt_profile.py',
    'columbia_plume/roms_forcing.py',
    'columbia_plume/test_bathy_smoothing.py',
    'columbia_plume/tidal_forcing.py',
    'columbia_plume/timeseries_forcing.py',
    'dome/diagnostics.py',
    'dome/dome_setup.py',
    'dome/plot_histogram.py',
    'katophillips/plot_results.py',
    'lockExchange/diagnostics.py',
    'lockExchange/plotting.py',
    'lockExchange/submitRuns.py',
    'sediment_trench_2d/trench_hydro.py',
    'sediment_meander_2d/meander_hydro.py',
    'tidalfarm/tidalfarm.py',
    'tidal_barrage/plotting.py',
]

cwd = os.path.abspath(os.path.dirname(__file__))
examples_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'examples'))

exclude_files = [os.path.join(examples_dir, f) for f in exclude_files]

all_examples = glob.glob(os.path.join(examples_dir, '*/*.py'))
all_examples = [f for f in all_examples if f not in exclude_files]


@pytest.fixture(params=all_examples,
                ids=lambda x: os.path.basename(x))
def example_file(request):
    return os.path.abspath(request.param)


def test_examples(example_file, tmpdir, monkeypatch):
    assert os.path.isfile(example_file), 'File not found {:}'.format(example_file)
    # copy mesh files
    source = os.path.dirname(example_file)
    for f in glob.glob(os.path.join(source, '*.msh')):
        shutil.copy(f, str(tmpdir))
    # change workdir to temporary dir
    monkeypatch.chdir(tmpdir)
    subprocess.check_call([sys.executable, example_file])
