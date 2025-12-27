import numpy as np
from sklearn.preprocessing import StandardScaler
from config import *
from data_loader import load_eeg
from feature_pipeline import extract_features
from model_seq2seq import build_seq2seq
from explainability import permutation_importance
from visualizations import *

data, sfreq = load_eeg()

win = int(WINDOW_SEC * sfreq)
step = int(win * (1 - OVERLAP))

X = []
for start in range(0, data.shape[1] - win, step):
    X.append(extract_features(data[:,start:start+win], sfreq))

X = np.array(X)
X = StandardScaler().fit_transform(X)

Xs, Ys = [], []
for i in range(len(X) - SEQ_LEN):
    Xs.append(X[i:i+SEQ_LEN])
    Ys.append(X[i+1:i+SEQ_LEN+1])

Xs = np.array(Xs)
Ys = np.array(Ys)

model = build_seq2seq(Xs.shape[2])
history = model.fit([Xs,Xs], Ys, epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save(MODEL_NAME)

plot_loss(history)
plot_eeg(data)
plot_psd(data, sfreq)

scores = permutation_importance(model, Xs, Ys)
plot_importance(scores)
