import os
from io import BytesIO
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import onnxruntime as ort
from einops import rearrange
import matplotlib.pyplot as plt

# Optional Google Generative AI (agentic)
try:
    import google.generativeai as genai
    HAS_GENAI = True
except:
    HAS_GENAI = False


# ============================================================
#  🔐 API KEY (YOU MUST REPLACE THIS IN YOUR LOCAL CODE)
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"   # <-- paste your API key here manually


# ============================================================
# Streamlit UI settings
# ============================================================
st.set_page_config(page_title="EEG → ONNX → Agentic AI", layout="wide")
st.title("🧠 EEG → PSD → Encoder (ONNX/Fallback) → Decoder (ONNX/Fallback) → Agentic AI")
st.markdown("Upload a `.edf` file and run the entire EEG → PSD → Seq2Seq pipeline.")


# ============================================================
# Helper Functions
# ============================================================
def read_edf_bytes(file_bytes):
    raw = mne.io.read_raw_edf(BytesIO(file_bytes), preload=True, verbose=False)
    return raw.get_data(), raw.info["sfreq"], raw.info["ch_names"]


def compute_psd(eeg, sfreq, nperseg=512):
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, fs=sfreq, nperseg=nperseg)
        psd_list.append(pxx)
    return np.array(psd_list), f


# ---------- Fallback Models ----------
class FallbackEncoder:
    def __init__(self, seq_len=64, features_out=64):
        self.seq_len = seq_len
        self.features_out = features_out

    def predict(self, x):
        x = x[0]  # remove batch
        ts, ch = x.shape

        if ts < self.seq_len:
            pad = np.zeros((self.seq_len - ts, ch), dtype=np.float32)
            x = np.vstack([x, pad])
            ts = self.seq_len

        idx = np.linspace(0, ts - 1, self.seq_len).astype(int)
        seq = x[idx, :]

        rng = np.random.RandomState(42)
        W = rng.randn(ch, self.features_out).astype(np.float32)
        encoded = seq @ W

        return encoded[None, ...]


class FallbackDecoder:
    def __init__(self, vocab_size=30):
        self.vocab_size = vocab_size

    def predict(self, encoded):
        rng = np.random.RandomState(123)
        W = rng.randn(encoded.shape[2], self.vocab_size).astype(np.float32)
        logits = encoded @ W

        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
        return probs


# ---------- ONNX Inference ----------
def run_onnx(model_path, arr):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: arr.astype(np.float32)})
    return out[0]


# ============================================================
# MAIN UI
# ============================================================
uploaded = st.file_uploader("Upload EEG (.edf)", type=["edf"])

if uploaded:
    st.info("Reading EEG... Please wait.")
    eeg_data, sfreq, ch_names = read_edf_bytes(uploaded.read())
    st.success(f"Loaded {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples @ {sfreq} Hz")

    # PSD
    st.subheader("📊 Power Spectral Density")
    psd, freqs = compute_psd(eeg_data, sfreq)

    fig, ax = plt.subplots(figsize=(8, 3))
    for i in range(min(4, psd.shape[0])):
        ax.semilogy(freqs, psd[i], label=ch_names[i])
    ax.legend(fontsize="small")
    st.pyplot(fig)

    # Prepare model input
    psd_T = psd.T  # (freqs × channels)
    psd_T = (psd_T - psd_T.mean()) / (psd_T.std() + 1e-9)
    model_input = psd_T[None, ...].astype(np.float32)

    st.write("Model input shape:", model_input.shape)

    encoder_present = os.path.exists("encoder.onnx")
    decoder_present = os.path.exists("decoder.onnx")

    # ============================================================
    #  ONNX or fallback
    # ============================================================
    if encoder_present and decoder_present:
        st.success("Found ONNX models — running ONNX inference.")
        try:
            encoded = run_onnx("encoder.onnx", model_input)
            decoded = run_onnx("decoder.onnx", encoded)
        except Exception as e:
            st.error("ONNX inference failed, switching to fallback: " + str(e))
            encoder_present = decoder_present = False

    if not (encoder_present and decoder_present):
        st.warning("Using fallback encoder/decoder (demo mode)")
        encoder = FallbackEncoder()
        encoded = encoder.predict(model_input)
        decoder = FallbackDecoder(vocab_size=30)
        decoded = decoder.predict(encoded)

    # Token predictions
    tokens = np.argmax(decoded, axis=-1)[0]
    st.subheader("🔡 Decoded Tokens (First 40)")
    st.write(tokens[:40].tolist())

    # Channel importance
    st.subheader("📌 Channel Importance (Simple)")
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
    importance = np.zeros(psd.shape[0])

    for (low, high) in bands.values():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) > 0:
            importance += psd[:, idx].mean(axis=1)

    importance /= importance.sum()
    st.bar_chart(importance)
    st.json({ch_names[i]: float(importance[i]) for i in range(len(ch_names))})

    # ============================================================
    #  Agentic LLM Interpretation
    # ============================================================
    st.subheader("🤖 Agentic AI Interpretation")

    if HAS_GENAI:
        if st.button("Run Agentic Interpretation"):
            try:
                genai.configure(api_key=API_KEY)

                prompt = f"""
                You are an expert EEG interpreter. Based on token sequence and channel importance,
                explain brain-state meaning, anomalies, and recommendations.

                Tokens: {tokens[:40].tolist()}
                Channel importance: { {ch_names[i]: float(importance[i]) for i in range(len(ch_names))} }
                """

                model = genai.GenerativeModel("gemini-2.5-flash")
                resp = model.generate_content(prompt)
                st.write(resp.text)

            except Exception as e:
                st.error("Agentic AI failed: " + str(e))
    else:
        st.info("google-generativeai not installed — agentic AI disabled.")

else:
    st.info("Upload an EEG .edf file to begin.")
