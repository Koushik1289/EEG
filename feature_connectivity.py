import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_connectivity_features(window):
    corr = np.corrcoef(window)
    sim = cosine_similarity(window)
    return list(np.mean(corr, axis=0)) + list(np.mean(sim, axis=0))
