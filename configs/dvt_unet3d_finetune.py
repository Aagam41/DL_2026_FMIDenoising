"""
dvt_unet3d_finetune — config for per-stack fine-tuning at submission time
on T4.

Loads pretrained weights from /opt/ml/model/dvt_unet3d.pth (saved by
scripts/pretrain_dvt.py), then runs a short fine-tune on the input stack
before inference.

Architecture MUST match dvt_unet3d_pretrain.py exactly (same backbone,
DVT bottleneck, preprocessing) so the pretrained state_dict loads cleanly.

Expected timing on T4 at this config:
  Stage 0 (warmup):    0 iters (skipped — pretrained model already knows
                                  the data distribution)
  Stage 1 (n2v):       800 iters (~5-6 min)
  Inference:           ~30-50 s
  Total per stack:     ~6-7 min
  7 stacks:            ~45-50 min (well under 1 hour)

If pretrained init is missing, the inference.py wrapper falls back to
training from scratch, which is slow — so always check that
/opt/ml/model/dvt_unet3d.pth is present in your submission.
"""

CONFIG = {
    "algo":         "dvt_unet3d",
    "name":         "dvt_unet3d_finetune",
    "description":  "DVT fine-tune (load pretrained, short adapt, infer).",
    "paper_frame":  750,

    # ── Backbone — MUST match dvt_unet3d_pretrain.py ─────────────
    "base_ch":      64,

    # ── DVT bottleneck — MUST match dvt_unet3d_pretrain.py ──────
    "token_dim":    256,
    "grid_shape":   (4, 8, 8),
    "n_vit_blocks": 2,
    "n_heads":      4,

    # ── Patch sampling ────────────────────────────────────────────
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   2,            # T4 16GB; keep batch small

    # ── Schedule (short fine-tune from pretrained init) ──────────
    # No warmup — pretrained model already understands the data
    # distribution, so warmup against temporal median is redundant.
    "warmup_iters": 0,
    "n2v_iters":    800,
    # Lower LR than scratch — standard 5-10× reduction for fine-tuning
    "lr":           1e-4,

    # ── Noise2Void mask — same as pretrain ───────────────────────
    "mask_ratio":   0.023227574381284047,
    "mask_radius":  1,

    # ── Preprocessing — MUST match dvt_unet3d_pretrain.py ────────
    "normalization":   "p3_p97",
    "temporal_target": "full_stack_median_2d",
}
