"""
Algorithm registry. Every denoising algorithm in the project is listed
here with a short name, a module path, and metadata.

To switch which algo runs, change the `algo` field of your config — no
code changes required.

To add a NEW algorithm:
  1. Drop a new module into algos/ that exposes the uniform API:
       compute_norm_params, normalize, denormalize
       train_self_supervised(stack, device, config) -> (model, cfg)
       denoise_stack(model, stack, config, device)  -> np.ndarray
       save_checkpoint(model, config, path)
       load_checkpoint(path, device=None) -> (model, cfg)
  2. Add an entry to REGISTRY below.

The runner ONLY interacts with algos through this registry + the
uniform API. It never imports algo modules directly.
"""

import importlib
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AlgoSpec:
    name: str                       # short identifier used everywhere
    module: str                     # importable module path
    family: str                     # grouping label (for tables/plots)
    description: str                # one-line summary
    paper: str = ""                 # optional reference
    tags: list = field(default_factory=list)


REGISTRY: Dict[str, AlgoSpec] = {

    "n2v_unet3d": AlgoSpec(
        name="n2v_unet3d",
        module="algos.n2v_unet3d",
        family="UNet",
        description="Standard 3D U-Net trained with Noise2Void blind-spot loss.",
        paper="Krull et al., CVPR 2019",
        tags=["unet", "n2v"],
    ),

    "n2v_unet3d_biasfree": AlgoSpec(
        name="n2v_unet3d_biasfree",
        module="algos.n2v_unet3d_biasfree",
        family="UNet",
        description="Bias-free 3D U-Net for Noise2Void (Mohan et al. ICLR 2019).",
        paper="Mohan et al., ICLR 2019",
        tags=["unet", "n2v", "bias-free"],
    ),

    "dvt_unet3d": AlgoSpec(
        name="dvt_unet3d",
        module="algos.dvt_unet3d",
        family="DVT",
        description="3D U-Net with DVT-style transformer bottleneck.",
        paper="Yang et al., 2024 (arXiv 2401.02957)",
        tags=["unet", "transformer", "dvt"],
    ),

    "restormer3d": AlgoSpec(
        name="restormer3d",
        module="algos.restormer3d",
        family="Restormer",
        description="3D adaptation of Restormer (MDTA + GDFN) for denoising.",
        paper="Zamir et al., CVPR 2022 (arXiv 2111.09881)",
        tags=["transformer", "mdta", "gdfn"],
    ),

    "restormer3d_v2": AlgoSpec(
        name="restormer3d_v2",
        module="algos.restormer3d_v2",
        family="Restormer",
        description="Restormer3D v2: multi-channel priors + hybrid loss + EMA + TTA.",
        paper="Zamir et al., CVPR 2022",
        tags=["transformer", "mdta", "gdfn", "ema", "tta"],
    ),

    "swin_unet3d": AlgoSpec(
        name="swin_unet3d",
        module="algos.swin_unet3d",
        family="Swin",
        description="3D Swin-Unet with GSC, FUE, and Restormer-style decoder.",
        paper="Cao et al., ECCV 2022; Xing et al., MICCAI 2024",
        tags=["swin", "transformer", "gsc", "fue"],
    ),

    "fm2s_dvt": AlgoSpec(
        name="fm2s_dvt",
        module="algos.fm2s_dvt",
        family="FM2S",
        description="FM2S 2D CNN spatial path + DVT-style Temporal-ViT refiner.",
        paper="Wang 2024 + Yang 2024",
        tags=["fm2s", "temporal-vit", "dvt"],
    ),

    "fm2s_classic": AlgoSpec(
        name="fm2s_classic",
        module="algos.fm2s_classic",
        family="FM2S",
        description="Plain FM2S (paper recipe, video mode with 8-fold self-ensemble).",
        paper="Wang et al., 2024 (arXiv 2412.10031)",
        tags=["fm2s", "cnn"],
    ),

    "n2v_3d_chhayansh": AlgoSpec(
        name="n2v_3d_chhayansh",
        module="algos.n2v_3d_chhayansh",
        family="UNet",
        description="2-stage 3D U-Net + BatchNorm trained with N2V; ships pre-trained weights from the AI4Life CIDC 2025 community submission.",
        paper="chhayanshporwal/3d-n2v-calcium-denoising (community submission, 2025)",
        tags=["unet", "n2v", "pretrained", "baseline"],
    ),

    "nafnet2d": AlgoSpec(
        name="nafnet2d",
        module="algos.nafnet2d",
        family="NAFNet",
        description="Per-frame NAFNet (SimpleGate + SCA) trained with 2D Noise2Void.",
        paper="Chen et al., ECCV 2022 (arXiv 2204.04676)",
        tags=["nafnet", "n2v", "2d"],
    ),

    "nafnet3d": AlgoSpec(
        name="nafnet3d",
        module="algos.nafnet3d",
        family="NAFNet",
        description="3D NAFNet (Conv3d throughout) trained with 3D Noise2Void; volumetric extension of the ECCV 2022 architecture.",
        paper="Chen et al., ECCV 2022 (extended to 3D)",
        tags=["nafnet", "n2v", "3d"],
    ),

    "deepcad": AlgoSpec(
        name="deepcad",
        module="algos.deepcad",
        family="DeepCAD",
        description="DeepCAD 3D U-Net (Li et al. Nat Methods 2021). Architecture-faithful; trained with our N2V flow rather than the official interleaved-frames N2N scheme.",
        paper="Li et al., Nature Methods 2021",
        tags=["deepcad", "unet", "n2v", "3d"],
    ),

    "swinir2d": AlgoSpec(
        name="swinir2d",
        module="algos.swinir2d",
        family="SwinIR",
        description="SwinIR for per-frame denoising — paper architecture (Liang et al. ICCVW 2021), denoising configuration (no upscaling head).",
        paper="Liang et al., ICCVW 2021 (arXiv 2108.10257)",
        tags=["swinir", "swin", "transformer", "2d"],
    ),

    "swinir3d": AlgoSpec(
        name="swinir3d",
        module="algos.swinir3d",
        family="SwinIR",
        description="SwinIR extended to 3D windowed attention (our research extension — NOT from any paper).",
        paper="Liang et al., ICCVW 2021 (extended to 3D)",
        tags=["swinir", "swin", "transformer", "3d"],
    ),

    "srdtrans": AlgoSpec(
        name="srdtrans",
        module="algos.srdtrans",
        family="SRDTrans",
        description="SRDTrans architecture (Li et al. Nat Comp Sci 2023). Lightweight spatiotemporal transformer with temporal encoder/decoder; trained with our N2V flow rather than the official spatial-redundancy sampling.",
        paper="Li et al., Nature Computational Science 2023",
        tags=["srdtrans", "transformer", "3d"],
    ),
}


def get_algo(name: str):
    """Import and return the algo module for `name`."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: "
            f"{sorted(REGISTRY.keys())}"
        )
    spec = REGISTRY[name]
    return importlib.import_module(spec.module)


def get_spec(name: str) -> AlgoSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: "
            f"{sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def list_algos():
    """Return the registry as a list of dicts (for CSV writing)."""
    return [
        {
            "name": s.name,
            "module": s.module,
            "family": s.family,
            "description": s.description,
            "paper": s.paper,
            "tags": ";".join(s.tags),
        }
        for s in REGISTRY.values()
    ]
