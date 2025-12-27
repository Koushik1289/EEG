import os
import time
import urllib.request
import numpy as np
import tensorflow as tf
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "openneuro_data"
EEG_FILE = os.path.join(DATA_DIR, "sub-01_eeg.edf")
EEG_URL = "https://s3.amazonaws.com/openneuro.org/ds002778/sub-01/eeg/sub-01_task-rest_eeg.edf"
MODEL_NAME = "eeg_model.h5"

WINDOW_SEC = 1.0
OVERLAP = 0.75
SEQ_LEN = 12
EPOCHS = 250
BATCH_SIZE = 64
LR = 1e-4

def log(x):
    print(f"[{time.strftime('%H:%M:%S')}] {x}")

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(EEG_FILE):
    urllib.request.urlretrieve(EEG_URL, EEG_FILE)

raw = mne.io.read_raw_edf(EEG_FILE, preload=True, verbose=False)
raw.pick_types(eeg=True)
raw.filter(0.5, 40.0)

data = raw.get_data()
sfreq = raw.info["sfreq"]
channels = data.shape[0]

def psd_features(sig):
    freqs, psd = welch(sig, fs=sfreq, nperseg=256)
    d = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
    t = np.mean(psd[(freqs >= 4) & (freqs < 8)])
    a = np.mean(psd[(freqs >= 8) & (freqs < 13)])
    b = np.mean(psd[(freqs >= 13) & (freqs < 30)])
    g = np.mean(psd[(freqs >= 30) & (freqs < 40)])
    tot = d + t + a + b + g + 1e-8
    return [d, t, a, b, g, d/tot, t/tot, a/tot, b/tot, g/tot]

def time_features(sig):
    return [
        np.mean(sig),
        np.std(sig),
        np.var(sig),
        np.max(sig) - np.min(sig),
        np.sqrt(np.mean(sig**2)),
        skew(sig),
        kurtosis(sig)
    ]

def nonlinear_features(sig):
    diff = np.diff(sig)
    return [
        np.mean(np.abs(diff)),
        np.std(diff),
        np.mean(diff**2)
    ]

def channel_features(sig):
    f = []
    f.extend(time_features(sig))
    f.extend(psd_features(sig))
    f.extend(nonlinear_features(sig))
    return f

win = int(WINDOW_SEC * sfreq)
step = int(win * (1 - OVERLAP))

X = []
for start in range(0, data.shape[1] - win, step):
    window = data[:, start:start+win]
    vec = []
    for ch in window:
        vec.extend(channel_features(ch))
    corr = np.corrcoef(window)
    sim = cosine_similarity(window)
    vec.extend(np.mean(corr, axis=0))
    vec.extend(np.mean(sim, axis=0))
    X.append(vec)

X = np.array(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_seq = []
Y_seq = []
for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
    Y_seq.append(X[i+1:i+SEQ_LEN+1])

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

enc_in = tf.keras.Input(shape=(SEQ_LEN, X_seq.shape[2]))
l1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(enc_in)
l2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l1)
l3 = tf.keras.layers.Add()([l1, l2])
enc_out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128, return_state=True)
)(l3)

h = tf.keras.layers.Concatenate()([fh, bh])
c = tf.keras.layers.Concatenate()([fc, bc])

dec_in = tf.keras.Input(shape=(SEQ_LEN, X_seq.shape[2]))
dec_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
dec_out, _, _ = dec_lstm(dec_in, initial_state=[h, c])
dec_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X_seq.shape[2]))
dec_out = dec_dense(dec_out)

model = tf.keras.Model([enc_in, dec_in], dec_out)
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")

history = model.fit([X_seq, X_seq], Y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

model.save(MODEL_NAME)

plt.figure(figsize=(8,4))
plt.plot(history.history["loss"])
plt.show()

plt.figure(figsize=(10,4))
for i in range(min(6, channels)):
    plt.plot(data[i][:2000] + i*50)
plt.show()

for i in range(3):
    freqs, psd = welch(data[i], fs=sfreq, nperseg=512)
    plt.plot(freqs, psd)
plt.show()

encoder = tf.keras.Model(enc_in, enc_out)
latent = encoder.predict(X_seq)

pca = PCA(n_components=3)
latent_pca = pca.fit_transform(latent)

plt.figure(figsize=(6,6))
plt.scatter(latent_pca[:,0], latent_pca[:,1], s=5)
plt.show()

baseline = mean_squared_error(
    Y_seq.reshape(-1, Y_seq.shape[-1]),
    model.predict([X_seq, X_seq]).reshape(-1, Y_seq.shape[-1])
)

importance = []
for i in range(X_seq.shape[2]):
    Xp = X_seq.copy()
    np.random.shuffle(Xp[:,:,i])
    loss = mean_squared_error(
        Y_seq.reshape(-1, Y_seq.shape[-1]),
        model.predict([Xp, Xp]).reshape(-1, Y_seq.shape[-1])
    )
    importance.append(loss - baseline)

plt.figure(figsize=(10,4))
plt.plot(importance)
plt.show()

print("saved eeg_model.h5")
