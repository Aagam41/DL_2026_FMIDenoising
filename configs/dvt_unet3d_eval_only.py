"""
dvt_unet3d_eval_only — config for evaluating a pretrained DVT checkpoint
WITHOUT any per-stack training.

Use with `--pretrained <ckpt.pth>` to load the checkpoint and run inference
directly. Both `warmup_iters` and `n2v_iters` are 0, so the training loop
is a no-op and the loaded weights are used as-is for inference.

Architecture MUST match the pretrained checkpoint's config (and the
fine-tune config) — see dvt_unet3d_pretrain.py.

Expected timing on T4:
  Train:       0 s (skipped entirely)
  Inference:  ~30-50 s/stack
  Total:      ~30-50 s/stack (the absolute floor — no per-stack adaptation)

This is mode A in the pretraining workflow: zero adaptation to the test
stack's specific noise, but maximum speed. Use it to benchmark the
pretrained model in isolation, or as a fallback when even 800 iters of
fine-tuning is too slow.

Compare against dvt_unet3d_finetune.py to see how much the 800-iter
fine-tune helps on YOUR data.
"""

CONFIG = {
    "algo":         "dvt_unet3d",
    "name":         "dvt_unet3d_eval_only",
    "description":  "DVT eval only (load pretrained, skip training, infer).",
    "paper_frame":  750,

    # ── Backbone — MUST match dvt_unet3d_pretrain.py ─────────────
    "base_ch":      64,

    # ── DVT bottleneck — MUST match dvt_unet3d_pretrain.py ──────
    "token_dim":    256,
    "grid_shape":   (4, 8, 8),
    "n_vit_blocks": 2,
    "n_heads":      4,

    # ── Patch sampling (used at inference only) ──────────────────
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   2,

    # ── Schedule: ZERO TRAINING ──────────────────────────────────
    # Both stages are no-ops. The model uses whatever weights it was
    # initialized with (the pretrained --pretrained checkpoint).
    "warmup_iters": 0,
    "n2v_iters":    0,
    # LR doesn't matter when iters=0, but kept for config completeness
    "lr":           1e-4,

    # ── Noise2Void mask (unused when n2v_iters=0) ───────────────
    "mask_ratio":   0.023227574381284047,
    "mask_radius":  1,

    # ── Preprocessing — MUST match dvt_unet3d_pretrain.py ────────
    "normalization":   "p3_p97",
    "temporal_target": "full_stack_median_2d",
}
