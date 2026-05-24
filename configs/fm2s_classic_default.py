"""fm2s_classic — plain FM2S (paper recipe, video mode, 8-fold self-ensemble)."""

CONFIG = {
    "algo":         "fm2s_classic",
    "name":         "fm2s_classic_default",
    "description":  "FM2S paper recipe in video mode.",
    "paper_frame":  750,

    "n_chan":       5,
    "train_num":    450,
    "max_epoch":    10,
    "stage1_steps": 10,
    "lr":           1e-3,
}
