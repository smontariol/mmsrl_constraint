cyclic_subsample_shorter = [
        {"subsample_labels": "macro"},
        {"subsample_labels": "micro"},
]

cyclic_subsample = [
        {"subsample_labels": "macro"},
        {"subsample_labels": "affine", "frequency_factor": 0.1, "frequency_bias": 0.05},
        {"subsample_labels": "micro"},
        {"subsample_labels": "affine", "frequency_factor": 0.7, "frequency_bias": 0.05},
]

multilabel = False
subsample_labels = None
