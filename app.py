# app.py
# Streamlit app: EEG -> PSD -> (ONNX or fallback) -> use Gemini to produce BOTH
#  - ground-truth estimate (from Gemini)
#  - predicted transcription (from Gemini)
# Then compare them (char/word accuracy), show diffs & graphs.
#
# Requirements (in requirements.txt):
# streamlit, numpy, scipy, mne, onnxruntime, einops, matplotlib, google-generativeai, protobuf>=4.23.4
#
# SECURITY: Do NOT hardcode your API key. Put it in Streamlit Secrets or an environment var:
# st.secrets["GOOGLE_API_KEY"] = "..." or export GOOGLE_API_KEY=...

import os
from io import BytesIO
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import onnxruntime as ort
import matplotlib.pyplot as plt
import random

# Try import Gemini client
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# ---------------- CONFIG ----------------
# Local placeholder (ONLY use locally, do not commit real keys)
API_KEY = "AIzaSyAtX1QJdv-y5xasT3elZ-fqQiPZUT8kwpY" # optional fallback to env
# You can also rely on st.secrets["GOOGLE_API_KEY"] in Streamlit Cloud

FALLBACK_VOCAB = list("abcdefghijklmnopqrstuvwxyz ')-,.") + [" "]

st.set_page_config(page_title="EEG → Gemini (GT & Pred) → Compare", layout="wide")
st.title("EEG → Gemini (Estimate ground-truth & Prediction) → Comparison")
st.markdown(
    "This app uses Gemini to produce both a *ground-truth estimate* and a *predicted transcription* from EEG features. "
    "Gemini is called only if a valid `GOOGLE_API_KEY` is available in Streamlit secrets or env vars. "
    "If Gemini is not available, the app falls back to ONNX/fallback decoder."
)

# ---------------- Helpers ----------------
def get_gemini_api_key():
    # Check for secrets (recommended) then env then local placeholder
    key = None
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", None)
    if not key:
        key = API_KEY if API_KEY and API_KEY != "AIzaSyAtX1QJdv-y5xasT3elZ-fqQiPZUT8kwpY" else None
    return key

def configure_gemini_or_raise():
    key = get_gemini_api_key()
    if not key:
        raise RuntimeError("No Google API key found. Set st.secrets['GOOGLE_API_KEY'] or environment variable GOOGLE_API_KEY.")
    genai.configure(api_key=key)

def gemini_generate(prompt: str, model_name="gemini-1.5", temperature=0.1):
    # wrapper for genai generate content
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai package not installed.")
    # configure on each call to ensure key available
    configure_gemini_or_raise()
    # Use the modern generate_content API if available
    if hasattr(genai, "GenerativeModel"):
        m = genai.GenerativeModel(model_name)
        out = m.generate_content(prompt, temperature=temperature)
        # response text usually in .text
        return out.text if hasattr(out, "text") else str(out)
    else:
        # older client fallback
        out = genai.generate(prompt=prompt, temperature=temperature)
        # older clients return dict-like object
        return out.get("candidates", [{}])[0].get("content", "") if isinstance(out, dict) else str(out)

def read_edf_bytes(file_bytes):
    raw = mne.io.read_raw_edf(BytesIO(file_bytes), preload=True, verbose=False)
    return raw.get_data(), raw.info["sfreq"], raw.info["ch_names"]

def compute_psd(eeg, sfreq, nperseg=512):
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, fs=sfreq, nperseg=nperseg)
        psd_list.append(pxx)
    return np.array(psd_list), f

# ONNX/fallback helpers (keeps app working without Gemini)
class FallbackEncoder:
    def __init__(self, seq_len=64, features_out=64):
        self.seq_len = seq_len
        self.features_out = features_out
    def predict(self, x):
        x = x[0].astype(np.float32)
        ts, ch = x.shape
        if ts < self.seq_len:
            pad = np.zeros((self.seq_len - ts, ch), dtype=np.float32)
            x = np.vstack([x, pad])
            ts = self.seq_len
        idx = np.linspace(0, ts - 1, self.seq_len).astype(int)
        seq = x[idx, :]
        rng = np.random.RandomState(42)
        W = rng.randn(seq.shape[1], self.features_out).astype(np.float32)
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
        probs = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
        return probs

def run_onnx(model_path, arr):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: arr.astype(np.float32)})
    return out[0] if isinstance(out, (list, tuple)) else out

# Text utilities
def tokens_to_text(tokens, vocab):
    chars = []
    for t in tokens:
        idx = int(t) % len(vocab)
        chars.append(vocab[idx])
    text = "".join(chars)
    return " ".join(text.split()).strip()

def normalize_text(s):
    return " ".join(s.lower().strip().split())

def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def char_accuracy(pred, true):
    if len(true) == 0:
        return 0.0 if len(pred) > 0 else 100.0
    ed = levenshtein(pred, true)
    acc = max(0.0, 1.0 - ed / max(len(true), 1))
    return acc * 100.0

def word_accuracy(pred, true):
    if len(true.strip()) == 0:
        return 0.0 if len(pred.strip()) > 0 else 100.0
    p_words = pred.split()
    t_words = true.split()
    minl = min(len(p_words), len(t_words))
    matches = sum(1 for i in range(minl) if p_words[i] == t_words[i])
    total = max(len(t_words), 1)
    return (matches / total) * 100.0

# ---------------- UI ----------------
st.sidebar.markdown("## Config & Gemini status")
key = get_gemini_api_key()
if key:
    st.sidebar.success("GOOGLE_API_KEY available")
else:
    st.sidebar.warning("No GOOGLE_API_KEY found — Gemini disabled (falls back to ONNX/fallback).")

uploaded = st.file_uploader("Upload .edf EEG file", type=["edf"])
st.markdown("---")
st.markdown("### Gemini-driven ground-truth & prediction mode")
st.write(
    "If Gemini key is available, the app will ask Gemini to **estimate a ground-truth transcription** from EEG features "
    "and also to **produce a predicted transcription**. Both texts are compared. "
    "This is useful for research/demo; treat Gemini outputs as *estimates*, not true labels."
)

# Optional manual ground-truth (user can still paste if desired, but we'll prefer Gemini-provided GT when enabled)
user_gt = st.text_area("Manual ground-truth (optional) — if left empty and Gemini is used, Gemini will provide the GT", height=120)

# Controls for Gemini behavior
use_gemini_for_text = st.checkbox("Use Gemini to generate ground-truth & prediction (requires key)", value=True if key else False)
gemini_model = st.selectbox("Gemini model", options=["gemini-2.5", "gemini-2.5-flash"], index=0)
temperature = st.slider("Generation temperature (lower = deterministic)", 0.0, 1.0, 0.1, 0.05)

if uploaded is None:
    st.info("Upload an EDF file to start.")
    st.stop()

# Read EEG and compute PSD
try:
    st.info("Reading EDF...")
    eeg_data, sfreq, ch_names = read_edf_bytes(uploaded.read())
    st.success(f"Loaded EEG — {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples @ {sfreq} Hz")
except Exception as e:
    st.error(f"Failed to read EDF: {e}")
    st.stop()

st.subheader("Power Spectral Density (example channels)")
psd, freqs = compute_psd(eeg_data, sfreq)
fig, ax = plt.subplots(figsize=(9, 3))
for i in range(min(4, psd.shape[0])):
    ax.semilogy(freqs, psd[i], label=ch_names[i])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend(fontsize="small")
st.pyplot(fig)

# Prepare model_input (freqs x channels => sequence)
psd_T = psd.T
psd_T = (psd_T - psd_T.mean()) / (psd_T.std() + 1e-9)
model_input = psd_T[None, ...].astype(np.float32)
st.write("Model input shape:", model_input.shape)

# Run ONNX/fallback encoder to get encoded features (these are used to create prompt summaries)
encoder_onx = "encoder.onnx"
decoder_onx = "decoder.onnx"
encoded = None
decoded_probs = None
use_onnx = os.path.exists(encoder_onx) and os.path.exists(decoder_onx)

if use_onnx:
    try:
        st.info("Running ONNX encoder to extract features for Gemini prompt...")
        encoded = run_onnx(encoder_onx, model_input)
        st.success("ONNX encoder ran successfully.")
    except Exception as e:
        st.warning(f"ONNX encoder failed; using fallback encoder for features. ({e})")
        use_onnx = False

if not use_onnx:
    encoder = FallbackEncoder(seq_len=64, features_out=64)
    encoded = encoder.predict(model_input)

# Build a compact summary of features for Gemini prompt:
# We'll report: number of channels, sampling rate, top frequency peaks for first few channels, and channel importance by band
def summarize_psd_for_prompt(psd, freqs, ch_names, top_n_channels=4):
    # compute band power per channel for common bands
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
    band_power = {}
    for b, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if idx.size:
            band_power[b] = psd[:, idx].mean(axis=1).tolist()
        else:
            band_power[b] = [0.0] * psd.shape[0]
    # top channels by average power across bands
    avg_power = np.array([np.mean([band_power[b][i] for b in bands]) for i in range(psd.shape[0])])
    top_idx = np.argsort(avg_power)[-top_n_channels:][::-1]
    top_info = []
    for idx in top_idx:
        # find frequency of max PSD for this channel
        peak_idx = np.argmax(psd[idx])
        peak_freq = float(freqs[peak_idx])
        top_info.append({"channel": ch_names[idx], "peak_freq": peak_freq, "avg_power": float(avg_power[idx])})
    return {"bands": {b: v[:6] for b, v in band_power.items()}, "top_channels": top_info}

psd_summary = summarize_psd_for_prompt(psd, freqs, ch_names)

# If Gemini is selected, call Gemini to produce BOTH ground-truth estimate and predicted transcription
gt_text_gemini = None
pred_text_gemini = None
pred_text_fallback = None

if use_gemini_for_text and HAS_GENAI and get_gemini_api_key():
    st.info("Gemini mode ON — generating ground-truth estimate & prediction from Gemini...")
    try:
        configure_gemini_or_raise()
        # Build prompt for ground-truth estimate
        prompt_gt = f"""
You are an EEG-to-text expert. Given the following EEG PSD summary and channel importance, produce the most likely
single short transcription (1-2 sentences) representing what the subject intended to say or think. Be concise and
produce plain text only (no explanations).

PSD summary (bands truncated): {psd_summary['bands']}
Top channels (name, peak frequency, avg_power): {psd_summary['top_channels']}
Sampling rate: {sfreq} Hz
Number of channels: {len(ch_names)}
Provide only the estimated ground-truth transcription on one line.
"""
        # Build prompt for prediction (slightly different; ask for a predicted transcription)
        prompt_pred = f"""
You are an EEG-to-text predictor. Given the same PSD summary and top channels, produce a predicted transcription
(what a decoding model might output), which can be noisier or shorter than the ground-truth. Keep it concise,
1-2 sentences, plain text only.
PSD summary: {psd_summary['bands']}
Top channels: {psd_summary['top_channels']}
Sampling rate: {sfreq} Hz
Number of channels: {len(ch_names)}
Provide only the predicted transcription on one line.
"""

        # Generate ground-truth estimate
        out_gt = gemini_generate(prompt_gt, model_name=gemini_model, temperature=temperature)
        gt_text_gemini = out_gt.strip().split("\n")[0].strip()

        # Generate predicted transcription
        out_pred = gemini_generate(prompt_pred, model_name=gemini_model, temperature=temperature + 0.05)
        pred_text_gemini = out_pred.strip().split("\n")[0].strip()

        st.success("Gemini generation completed.")
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        st.info("Falling back to manual or model-based predicted text.")
        gt_text_gemini = None
        pred_text_gemini = None

else:
    if use_gemini_for_text:
        st.warning("Gemini requested but not available (missing package or API key). Falling back to ONNX/fallback prediction.")
    else:
        st.info("Gemini generation disabled — using ONNX/fallback for predicted text (if available).")

# If Gemini produced ground-truth and/or prediction, use those; else allow user manual gt or model-based prediction
final_ground_truth = None
final_prediction = None

# Priority:
# 1) If Gemini produced GT, use it as final ground-truth (but if user provided manual GT, prefer manual GT and also show Gemini GT)
# 2) If Gemini produced prediction, use it as final prediction
# 3) Else, use manual GT/prediction from ONNX/fallback decoder

# 1) final_ground_truth
if user_gt and user_gt.strip():
    final_ground_truth = user_gt.strip()
    gemini_gt_used = False
else:
    if gt_text_gemini:
        final_ground_truth = gt_text_gemini
        gemini_gt_used = True
    else:
        final_ground_truth = ""  # empty if none
        gemini_gt_used = False

# 2) final_prediction
if pred_text_gemini:
    final_prediction = pred_text_gemini
    gemini_pred_used = True
else:
    # attempt to get prediction from ONNX decoder if available (we didn't run decoder earlier)
    if os.path.exists(decoder_onx):
        try:
            # If ONNX decoder expects encoded features input, ensure shapes match; here we try best-effort
            st.info("Running ONNX decoder to produce predicted tokens (fallback path).")
            # If encoded from ONNX encoder exists, use it; otherwise use encoded from fallback encoder
            decoded_probs = run_onnx(decoder_onx, encoded)
            predicted_tokens = np.argmax(decoded_probs, axis=-1)[0].tolist()
            final_prediction = tokens_to_text(predicted_tokens, FALLBACK_VOCAB)
            gemini_pred_used = False
        except Exception:
            gemini_pred_used = False
            final_prediction = tokens_to_text(np.argmax(FallbackDecoder(vocab_size=30).predict(encoded), axis=-1)[0].tolist(), FALLBACK_VOCAB)
    else:
        gemini_pred_used = False
        # fallback decoder
        final_prediction = tokens_to_text(np.argmax(FallbackDecoder(vocab_size=30).predict(encoded), axis=-1)[0].tolist(), FALLBACK_VOCAB)

# Show results: Gemini GT (if available), manual GT (if provided), and final prediction
st.subheader("Ground-truth (manual or Gemini) and Prediction")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Ground-truth (final)**")
    if gemini_gt_used:
        st.info("Ground-truth estimated by Gemini")
    else:
        st.info("Ground-truth provided manually (if any) or empty")
    st.text_area("Ground-truth text", value=final_ground_truth, height=160)
    st.download_button("Download Ground-truth", data=final_ground_truth, file_name="ground_truth.txt")
with col2:
    st.markdown("**Prediction (final)**")
    if gemini_pred_used:
        st.info("Prediction produced by Gemini")
    else:
        st.info("Prediction produced by model/fallback")
    st.text_area("Predicted text", value=final_prediction, height=160)
    st.download_button("Download Prediction", data=final_prediction, file_name="prediction.txt")

# Compute accuracies comparing final_prediction to final_ground_truth (if ground-truth empty, show info)
pred_norm = normalize_text(final_prediction)
gt_norm = normalize_text(final_ground_truth)

st.subheader("Accuracy & Comparison")
if not final_ground_truth or final_ground_truth.strip() == "":
    st.warning("No ground-truth available for numeric comparison. Either provide manual ground-truth in the text box or allow Gemini to estimate it.")
else:
    c_acc = char_accuracy(pred_norm, gt_norm)
    w_acc = word_accuracy(pred_norm, gt_norm)
    c1, c2 = st.columns(2)
    c1.metric("Character-level accuracy (%)", f"{c_acc:.2f}")
    c2.metric("Word-level accuracy (%)", f"{w_acc:.2f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Character", "Word"], [c_acc, w_acc], color=["#2b8cbe", "#fdae61"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    for i, v in enumerate([c_acc, w_acc]):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    st.pyplot(fig)

    # Inline character diff (first 600 chars)
    st.subheader("Character-level diff (highlights mismatches)")
    N = 600
    t_display = gt_norm[:N]
    p_display = pred_norm[:N]
    def make_char_diff_html(true_s, pred_s):
        L = max(len(true_s), len(pred_s))
        t_p = true_s.ljust(L)
        p_p = pred_s.ljust(L)
        html_true = []
        html_pred = []
        for i in range(L):
            ct = t_p[i]
            cp = p_p[i]
            if ct == cp:
                html_true.append(ct if ct != " " else "&middot;")
                html_pred.append(cp if cp != " " else "&middot;")
            else:
                html_true.append(f"<span style='background:#c6efce'>{ct if ct != ' ' else '&middot;'}</span>")
                html_pred.append(f"<span style='background:#ffc7ce'>{cp if cp != ' ' else '&middot;'}</span>")
        return (
            "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.4;'>True:  "
            + "".join(html_true)
            + "</div><div style='font-family:monospace;white-space:pre-wrap;line-height:1.4;'>Pred:  "
            + "".join(html_pred)
            + "</div>"
        )
    st.markdown(make_char_diff_html(t_display, p_display), unsafe_allow_html=True)

    # Word-level table
    st.subheader("Word-level comparison")
    p_words = pred_norm.split()
    t_words = gt_norm.split()
    max_len = max(len(p_words), len(t_words))
    md = "|#|Actual|Predicted|Match|\n|--:|--|--|--:|\n"
    for i in range(max_len):
        t = t_words[i] if i < len(t_words) else ""
        p = p_words[i] if i < len(p_words) else ""
        match = "✔" if t == p else "✖"
        md += f"|{i+1}|{t}|{p}|{match}|\n"
    st.markdown(md, unsafe_allow_html=True)

# Channel importance visualization (same as earlier)
st.subheader("Channel importance (band-power)")
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
ch_scores = np.zeros(psd.shape[0])
for (low, high) in bands.values():
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    if idx.size:
        ch_scores += psd[:, idx].mean(axis=1)
ch_scores = ch_scores / (ch_scores.sum() + 1e-12)
fig_ch, ax = plt.subplots(figsize=(9, 3))
ax.bar(range(len(ch_scores)), ch_scores)
ax.set_xticks(range(len(ch_scores)))
ax.set_xticklabels([c for c in ch_names], rotation=60, fontsize=8)
ax.set_ylabel("Relative importance")
st.pyplot(fig_ch)

# Agentic commentary: ask Gemini to interpret differences (optional)
if HAS_GENAI and get_gemini_api_key():
    if st.button("Ask Gemini to interpret the comparison (optional)"):
        try:
            configure_gemini_or_raise()
            prompt = (
                "You are an EEG analytics assistant. Given the ground-truth transcription and the predicted transcription "
                "derived from EEG PSD and channel importance, explain likely reasons for differences, whether prediction seems plausible, "
                "and suggest next steps to improve decoding. Provide a concise analysis.\n\n"
                f"Ground-truth: {final_ground_truth}\nPrediction: {final_prediction}\nPSD summary: {psd_summary['bands']}\nTop channels: {psd_summary['top_channels']}"
            )
            resp = gemini_generate(prompt, model_name=gemini_model, temperature=0.2)
            st.subheader("Gemini analysis")
            st.write(resp)
        except Exception as e:
            st.error(f"Gemini analysis failed: {e}")
else:
    st.info("Gemini analysis button requires google.generativeai installed and GOOGLE_API_KEY set.")

st.markdown("---")
st.caption("Notes: Gemini-produced 'ground-truth' is an estimate and not a substitute for human-labeled ground-truth. "
           "Use Gemini outputs for demo / research exploration only.")
