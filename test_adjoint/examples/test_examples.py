"""
Runs all adjoint example scripts. Only tests whether examples can be executed.
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

# list of all adjoint examples to run
adjoint_files = [
    'tidalfarm/tidalfarm.py',
    # 'channel_inversion/inverse_problem.py',  # FIXME requires obs time series
]

cwd = os.path.abspath(os.path.dirname(__file__))
examples_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'examples'))

include_files = [os.path.join(examples_dir, f) for f in adjoint_files]

all_examples = include_files


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
