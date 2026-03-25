from stanbkt.utils.compilation import (
    compile_stan_model,
    get_stan_model_cache_dir,
    clear_stan_cache,
    get_cache_root,
    list_cached_models,
    is_sys_windows,
)

# add RTools to PATH
if is_sys_windows():
    from cmdstanpy.utils import cxx_toolchain_path

    try:
        cxx_toolchain_path()  # adds RTools to PATH if found
        print("RTools found and added to PATH.")
    except ValueError as e:
        pass

__all__ = [
    "compile_stan_model",
    "get_stan_model_cache_dir",
    "clear_stan_cache",
    "get_cache_root",
    "list_cached_models",
]
