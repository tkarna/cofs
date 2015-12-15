import pytest


def pytest_addoption(parser):
    parser.addoption("--travis", action="store_true", default=False,
                     help="Only run tests marked for Travis")


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "not_travis: Mark a test that should not be run on Travis")


def pytest_runtest_setup(item):
    not_travis = item.get_marker("not_travis")
    if not_travis is not None and item.config.getoption("--travis"):
        pytest.skip("Skipping test marked not for Travis")


# Print a progress "." once a minute when running in travis mode
# This is an attempt to stop travis timing the builds out due to lack
# of output.
progress_process = None


def pytest_configure(config):
    global progress_process
    if config.getoption("--travis") and progress_process is None:
        import multiprocessing
        import py
        terminal = py.io.TerminalWriter()
        def writer():
            import time
            while True:
                terminal.write("still alive\n")
                time.sleep(60)
        progress_process = multiprocessing.Process(target=writer)
        progress_process.daemon = True
        progress_process.start()


def pytest_unconfigure(config):
    global progress_process
    if config.getoption("--travis") and progress_process is not None:
        progress_process.terminate()
