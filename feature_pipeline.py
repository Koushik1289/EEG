import numpy as np
from feature_time import extract_time_features
from feature_psd import extract_psd_features
from feature_nonlinear import extract_nonlinear_features
from feature_connectivity import extract_connectivity_features

def extract_features(window, sfreq):
    vec = []
    for ch in window:
        vec.extend(extract_time_features(ch))
        vec.extend(extract_psd_features(ch, sfreq))
        vec.extend(extract_nonlinear_features(ch))
    vec.extend(extract_connectivity_features(window))
    return np.array(vec)
