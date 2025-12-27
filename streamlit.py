import streamlit as st
import numpy as np
import tensorflow as tf
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="🧠 EEG-Driven Language Interface",
    layout="wide"
)

st.title("🧠 EEG-Driven Language Interface Showcase")

# ==========================================================
# PATH MAPPING (ONLY CHANGE THESE)
# ==========================================================
MODEL_PATH = "eeg_model.h5"                 # trained model path
EEG_PATH = "openneuro_data/sub-01_eeg.edf"  # EEG file path

# ==========================================================
# SAFETY CHECKS
# ==========================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at: {MODEL_PATH}")
    st.stop()

if not os.path.exists(EEG_PATH):
    st.error(f"EEG file not found at: {EEG_PATH}")
    st.stop()

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================================
# LOAD EEG
# ==========================================================
@st.cache_data
def load_eeg():
    raw = mne.io.read_raw_edf(EEG_PATH, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 40.0)
    return raw

raw = load_eeg()
data = raw.get_data()
sfreq = raw.info["sfreq"]

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(window, sfreq):
    feats = []
    for ch in window:
        feats.extend([
            np.mean(ch),
            np.std(ch),
            np.var(ch),
            np.max(ch) - np.min(ch),
            np.sqrt(np.mean(ch ** 2))
        ])
        freqs, psd = welch(ch, fs=sfreq, nperseg=256)
        d = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        t = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        a = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        b = np.mean(psd[(freqs >= 13) & (freqs < 30)])
        g = np.mean(psd[(freqs >= 30) & (freqs < 40)])
        tot = d + t + a + b + g + 1e-8
        feats.extend([d, t, a, b, g, d/tot, t/tot, a/tot, b/tot, g/tot])
    return np.array(feats)

# ==========================================================
# RAW EEG VISUALIZATION
# ==========================================================
st.subheader("📈 Raw EEG Signals")
fig, ax = plt.subplots(figsize=(10,4))
for i in range(min(6, data.shape[0])):
    ax.plot(data[i][:2000] + i * 50)
ax.set_title("Raw EEG (First 6 Channels)")
st.pyplot(fig)

# ==========================================================
# FEATURE MATRIX
# ==========================================================
WINDOW_SEC = 1.0
OVERLAP = 0.75

win = int(WINDOW_SEC * sfreq)
step = int(win * (1 - OVERLAP))

feature_vectors = []
for start in range(0, data.shape[1] - win, step):
    feature_vectors.append(
        extract_features(data[:, start:start + win], sfreq)
    )

X = np.array(feature_vectors)
X = StandardScaler().fit_transform(X)

# ==========================================================
# BUILD SEQUENCES
# ==========================================================
SEQ_LEN = 10
X_seq = []

for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i + SEQ_LEN])

X_seq = np.array(X_seq)

# ==========================================================
# MODEL PREDICTION
# ==========================================================
with st.spinner("Decoding EEG using trained model..."):
    predictions = model.predict([X_seq, X_seq])

latent_vector = np.mean(predictions, axis=(0, 1))

# ==========================================================
# EEG → TEXT DECODING
# ==========================================================
vocabulary = [
    "neural", "temporal", "signal", "pattern", "dynamic",
    "oscillation", "structure", "activity", "variation",
    "state", "sequence", "modulation", "synchrony",
    "network", "rhythm"
]

indices = np.abs(latent_vector)
indices = (indices / indices.max() * (len(vocabulary) - 1)).astype(int)

decoded_text = " ".join([vocabulary[i] for i in indices[:15]])

st.subheader("📝 Decoded EEG Text")
st.success(decoded_text)

# ==========================================================
# LATENT SPACE VISUALIZATION
# ==========================================================
st.subheader("🔍 Latent Space Projection")

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(predictions.reshape(predictions.shape[0], -1))

fig2, ax2 = plt.subplots()
ax2.scatter(latent_2d[:,0], latent_2d[:,1], s=5)
ax2.set_title("PCA Projection of Latent Space")
st.pyplot(fig2)

# ==========================================================
# MODEL SUMMARY
# ==========================================================
with st.expander("💻 Model Architecture"):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    st.text("\n".join(stringlist))
