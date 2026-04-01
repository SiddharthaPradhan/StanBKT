from stanbkt.utils.compilation import (
    compile_stan_model,
    get_stan_model_cache_dir,
    clear_stan_cache,
    get_cache_root,
    list_cached_models,
    is_sys_windows,
)
from stanbkt.utils.data_utils import ColumnNames, KCData, validate_data, format_kc_data
from stanbkt.utils.verbose import VerbosityLevel
from stanbkt.utils.sim import sim_simple_BKT
from stanbkt.utils.model_archive import pack_model_directory, unpack_model_archive

# add RTools to PATH
if is_sys_windows():
    from cmdstanpy.utils import cxx_toolchain_path

    try:
        cxx_toolchain_path()  # adds RTools to PATH if found
        print("RTools found and added to PATH.")
    except ValueError as e:
        pass

__all__ = [
    # Compilation
    "compile_stan_model",
    "get_stan_model_cache_dir",
    "clear_stan_cache",
    "get_cache_root",
    "list_cached_models",
    # Data utilities
    "ColumnNames",
    "KCData",
    "validate_data",
    "format_kc_data",
    # Verbosity
    "VerbosityLevel",
    # Model I/O
    "pack_model_directory",
    "unpack_model_archive",
    # Simulation
    "sim_simple_BKT",
]
