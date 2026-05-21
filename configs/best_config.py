"""
Best config found by HPO for restormer3d.

Best stSNR: 24.3582
HPO was run at reduced budget (--reduce 0.25); the iter counts
in this config are the FULL budget — use as a production config.
"""

CONFIG = {
    'algo'              : 'restormer3d',
    'paper_frame'       : 750,
    'patch_d'           : 32,
    'patch_hw'          : 64,
    'batch_size'        : 2,
    'bias_free'         : True,
    'ffn_expansion_factor': 2.0,
    'heads'             : (1, 2, 4, 8),
    'num_refinement_blocks': 2,
    'normalization'     : 'p3_p97',
    'temporal_target'   : 'temporal_median_2d',
    'temporal_overlap'  : 0.5,
    'lr'                : 0.0009883029120086638,
    'mask_ratio'        : 0.027043927794832567,
    'mask_radius'       : 2,
    'warmup_iters'      : 600,
    'n2v_iters'         : 4500,
    'dim'               : 32,
    'num_blocks'        : (1, 1, 1, 2),
}
