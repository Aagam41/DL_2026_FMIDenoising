"""
dvt_unet3d_default — re-export of the proven-best DVT config.

This file lets `scripts/benchmark.py` find the DVT config via its
standard `<algo>_default.py` lookup pattern. The actual CONFIG lives
in `dvt_unet3d_t4.py` — this is just a pointer to keep ONE source of
truth.
"""

# Load the canonical config from dvt_unet3d_t4.py at file-load time.
# We use exec rather than a relative import because configs/*.py are
# loaded by `configs.load_config` as standalone modules, not as part
# of the configs package, so relative imports raise ImportError.
import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_t4_path = _os.path.join(_here, "dvt_unet3d_t4.py")
_namespace = {}
with open(_t4_path) as _f:
    exec(compile(_f.read(), _t4_path, "exec"), _namespace)
CONFIG = dict(_namespace["CONFIG"])

# Tag the source so config.csv knows which file was the canonical one
CONFIG["__config_alias"] = "dvt_unet3d_default -> dvt_unet3d_t4"
