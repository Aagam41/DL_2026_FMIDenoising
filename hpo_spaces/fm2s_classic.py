"""HPO search space for fm2s_classic (paper recipe, video mode).

This is the original FM2S paper algorithm — fewer knobs than fm2s_dvt
since it's a single-stage CNN. Mostly schedule + lr.
"""

BASE_CONFIG = {
    "algo":            "fm2s_classic",
    "paper_frame":     750,
    "normalization":   "noop",   # FM2S handles its own scaling
    "temporal_target": "framework_default",
}

DESCRIPTION = (
    "FM2S classic — search over channel width, train_num, lr, and the "
    "two-stage step counts."
)


def suggest(trial):
    n_chan = trial.suggest_categorical("n_chan", [4, 5, 8])
    train_num = trial.suggest_int("train_num", 300, 700, step=50)
    max_epoch = trial.suggest_int("max_epoch", 6, 14)
    stage1_steps = trial.suggest_int("stage1_steps", 5, 20)
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    return {
        "n_chan":        n_chan,
        "train_num":     train_num,
        "max_epoch":     max_epoch,
        "stage1_steps":  stage1_steps,
        "lr":            lr,
    }
