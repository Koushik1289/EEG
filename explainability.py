import numpy as np
from sklearn.metrics import mean_squared_error

def permutation_importance(model, X, Y):
    baseline = mean_squared_error(
        Y.reshape(-1, Y.shape[-1]),
        model.predict([X,X]).reshape(-1, Y.shape[-1])
    )
    scores = []
    for i in range(X.shape[2]):
        Xp = X.copy()
        np.random.shuffle(Xp[:,:,i])
        loss = mean_squared_error(
            Y.reshape(-1, Y.shape[-1]),
            model.predict([Xp,Xp]).reshape(-1, Y.shape[-1])
        )
        scores.append(loss - baseline)
    return scores
