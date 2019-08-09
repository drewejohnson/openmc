import os
from pathlib import Path
from subprocess import Popen, STDOUT

import pytest
import pytest

from tests.regression_tests import config as regression_config


def pytest_addoption(parser):
    parser.addoption('--exe')
    parser.addoption('--mpi', action='store_true')
    parser.addoption('--mpiexec')
    parser.addoption('--mpi-np')
    parser.addoption('--update', action='store_true')
    parser.addoption('--build-inputs', action='store_true')


def pytest_configure(config):
    opts = ['exe', 'mpi', 'mpiexec', 'mpi_np', 'update', 'build_inputs']
    for opt in opts:
        if config.getoption(opt) is not None:
            regression_config[opt] = config.getoption(opt)


@pytest.fixture
def run_in_tmpdir(tmpdir):
    orig = tmpdir.chdir()
    try:
        yield
    finally:
        orig.chdir()


@pytest.fixture(scope="package")
def endf_data_path():
    endf_dir = os.environ['OPENMC_ENDF_DATA']
    endf_path = Path(endf_dir)
    # check for necessary directories
    for data_dir in ("neutrons", "thermal_scatt", "photoat", "atomic_relax",
                     "decay", "nfy"):
        assert (endf_path / data_dir).is_dir(), (
            "ENDF {} library not found".format(data_dir))
    return endf_path


@pytest.fixture(scope="package")
def needs_njoy():
    """Fixture to indicate that a test needs njoy"""
    try:
        proc = Popen(['njoy', '--version'], stderr=STDOUT)
        if proc.returncode == 0:
            return
    except FileNotFoundError:  # no njoy file found
        pass
    pytest.skip("njoy executable not found. Some openmc.data tests skipped")
