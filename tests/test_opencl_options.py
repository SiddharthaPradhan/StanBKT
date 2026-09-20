import pytest

from stanbkt.fits.fit_factory import FitFactory
from stanbkt.fits.fit_options import MCMCFitOptions
from stanbkt.fits.fit_types import FitMethod


def test_opencl_ids_require_stan_opencl():
    options = MCMCFitOptions(opencl_ids=(0, 0))
    with pytest.raises(ValueError, match="STAN_OPENCL"):
        FitFactory.verify_fit_options_compatibility(options, FitMethod.MCMC)


def test_opencl_ids_accepted_with_stan_opencl():
    options = MCMCFitOptions(opencl_ids=(0, 0))
    FitFactory.verify_fit_options_compatibility(
        options, FitMethod.MCMC, cpp_compile_kwargs={"STAN_OPENCL": True}
    )
    assert options.to_dict()["opencl_ids"] == (0, 0)
