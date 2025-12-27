import numpy as np

def extract_nonlinear_features(signal):
    diff = np.diff(signal)
    return [
        np.mean(np.abs(diff)),
        np.std(diff),
        np.mean(diff**2)
    ]
