"""
Best config found by HPO for fm2s_classic.

Best stSNR: 17.0552
HPO was run at reduced budget (--reduce 0.25); the iter counts
in this config are the FULL budget — use as a production config.
"""

CONFIG = {
    'algo'              : 'fm2s_classic',
    'paper_frame'       : 750,
    'normalization'     : 'noop',
    'temporal_target'   : 'framework_default',
    'n_chan'            : 4,
    'train_num'         : 500,
    'max_epoch'         : 14,
    'stage1_steps'      : 18,
    'lr'                : 0.0014230862751996227,
}
