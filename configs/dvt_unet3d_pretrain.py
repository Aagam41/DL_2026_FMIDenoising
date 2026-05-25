"""
dvt_unet3d_pretrain — config for offline pretraining of DVT-UNet3D
across multiple training stacks. Run on RTX 5090 (or any 24GB+ GPU).

This config is for `scripts/pretrain_dvt.py`. The architecture is identical
to dvt_unet3d_t4.py (the proven best-quality config) so the pretrained
checkpoint can be loaded into fine-tuning later without architecture
mismatches.

Differences from dvt_unet3d_t4.py:
  - batch_size 2 → 4   (5090 has 32GB VRAM)
  - warmup_iters/n2v_iters are placeholders; scripts/pretrain_dvt.py
    overrides them based on --total-iters and --warmup-iters CLI flags

Everything else (base_ch=64, token_dim=256, grid=(4,8,8), n_vit_blocks=2,
n_heads=4, patch=32×128×128, lr, mask params, normalization, temporal target)
matches the t4 best config exactly.
"""

CONFIG = {
    "algo":         "dvt_unet3d",
    "name":         "dvt_unet3d_pretrain",
    "description":  "DVT pretraining config (multi-stack on RTX 5090).",
    "paper_frame":  750,

    # ── Backbone — MUST match the fine-tune config ────────────────
    "base_ch":      64,

    # ── DVT bottleneck — MUST match the fine-tune config ──────────
    "token_dim":    256,
    "grid_shape":   (4, 8, 8),
    "n_vit_blocks": 2,
    "n_heads":      4,

    # ── Patch sampling ────────────────────────────────────────────
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   4,            # 5090 fits batch=4 at this patch size

    # ── Schedule — overridden by scripts/pretrain_dvt.py CLI ─────
    # These defaults are used only if the script doesn't override
    # (e.g. someone calls train_self_supervised directly with this config).
    "warmup_iters": 2000,
    "n2v_iters":    48000,
    "lr":           0.0006761147179060029,   # matches dvt_unet3d_t4

    # ── Noise2Void mask ──────────────────────────────────────────
    "mask_ratio":   0.023227574381284047,    # matches dvt_unet3d_t4
    "mask_radius":  1,                       # matches dvt_unet3d_t4

    # ── Preprocessing — MUST match the fine-tune config ──────────
    "normalization":   "p3_p97",
    "temporal_target": "full_stack_median_2d",
}
