"""Model package exports.

The heavy model modules are imported lazily so lightweight utilities can import
submodules such as ``models.aux_head`` without initializing optional dependencies.
"""

__all__ = ["LOGLO_FNO", "FNOWrapper", "UNet3D", "AuxHead"]


def __getattr__(name):
    if name == "LOGLO_FNO":
        from .loglo_fno import LOGLO_FNO

        return LOGLO_FNO
    if name == "FNOWrapper":
        from .fno_wrapper import FNOWrapper

        return FNOWrapper
    if name == "UNet3D":
        from .unet3d import UNet3D

        return UNet3D
    if name == "AuxHead":
        from .aux_head import AuxHead

        return AuxHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
