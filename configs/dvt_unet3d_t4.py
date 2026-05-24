"""
dvt_unet3d — DVT-style transformer bottleneck inside a 3D U-Net.

Best-quality config. Full fp32 training (no AMP, no GradScaler) — slower
than the old T4-budget variant but bulletproof against attention overflow
and numerical artifacts.

Expected timing:
    Training:  ~15 min per stack on T4-class GPU (or ~5 min on A100)
    Inference: ~50 s per stack
    Memory:    ~14 GB VRAM at patch=32×128×128, batch=2

Best F1 result on this config (your earlier run): stSNR=19.81, stPSNR=23.45.
"""

CONFIG = {
    "algo":         "dvt_unet3d",
    "name":         "dvt_unet3d_t4",
    "description":  "DVT-style bottleneck UNet — fp32, best quality.",
    "paper_frame":  750,

    # ── Backbone (~6.5M params) ───────────────────────────────────
    "base_ch":      64,

    # ── DVT bottleneck (256 tokens, 2 ViT blocks, 4 heads) ────────
    "token_dim":    256,
    "grid_shape":   (4, 8, 8),
    "n_vit_blocks": 2,
    "n_heads":      4,

    # ── Patch sampling ────────────────────────────────────────────
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   2,

    # ── Schedule ──────────────────────────────────────────────────
    "warmup_iters": 600,
    "n2v_iters":    5000,
    "lr":           0.0006761147179060029,

    # ── Noise2Void mask ──────────────────────────────────────────
    "mask_ratio":   0.023227574381284047,
    "mask_radius":  1,

    # ── Preprocessing ─────────────────────────────────────────────
    "normalization":   "p3_p97",
    "temporal_target": "full_stack_median_2d",
}
