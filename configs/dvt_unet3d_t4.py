"""
Best config found by HPO for dvt_unet3d.

Best stSNR: 22.9415
HPO was run at reduced budget (--reduce 0.25); the iter counts
in this config are the FULL budget — use as a production config.
"""

CONFIG = {
    'algo'              : 'dvt_unet3d',
    'paper_frame'       : 750,
    'patch_d'           : 32,
    'patch_hw'          : 128,
    'batch_size'        : 2,
    'n_vit_blocks'      : 2,
    'n_heads'           : 4,
    'grid_shape'        : (4, 8, 8),
    'normalization'     : 'p3_p97',
    'temporal_target'   : 'full_stack_median_2d',
    'lr'                : 0.0006761147179060029,
    'mask_ratio'        : 0.023227574381284047,
    'mask_radius'       : 1,
    'warmup_iters'      : 600,
    'n2v_iters'         : 5000,
    'base_ch'           : 64,
    'token_dim'         : 256,
}
