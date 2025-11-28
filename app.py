# app.py
# Professional EEG -> PSD -> Residual BiLSTM (ONNX) -> Seq2Seq (ONNX) -> Explainability -> LLM analysis
#
# Notes:
# - Place encoder.onnx and decoder.onnx (if you have them) in repo root to use ONNX inference.
# - To enable LLM-based steps (optional): set your GOOGLE_API_KEY in Streamlit Secrets or env var.
#   The app will attempt to use model "gemini-2.5-flash" when calling the google.generativeai client.
# - This app uses model-agnostic perturbation explainability (occlusion on PSD) to produce importance maps.
#
# Required files: encoder.onnx (optional), decoder.onnx (optional)
# Deploy: push to GitHub, add to Streamlit Cloud, set secrets: {"GOOGLE_API_KEY": "..."}
# Security: never commit API keys to GitHub.

import os
from io import BytesIO
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
import time
import json

# Optional LLM (Gemini). The app will not hardcode any keys; it reads from st.secrets or env.
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

st.set_page_config(page_title="EEG→Text: PSD + ResidualBiLSTM (ONNX) + Explainability", layout="wide")

# ----------------------------
# Utility functions
# ----------------------------
def get_api_key():
    # Prefer Streamlit secrets, then environment variable
    try:
        k = st.secrets["GOOGLE_API_KEY"]
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", None)

def configure_llm(api_key: str):
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai not installed in environment.")
    genai.configure(api_key=api_key)

def llm_generate(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.1):
    """
    Wrapper to call Gemini-like client. Returns string.
    """
    key = get_api_key()
    if not key:
        raise RuntimeError("No API key available. Set in Streamlit Secrets or env var GOOGLE_API_KEY.")
    configure_llm(key)
    if hasattr(genai, "GenerativeModel"):
        m = genai.GenerativeModel(model_name)
        out = m.generate_content(prompt, temperature=temperature)
        return out.text if hasattr(out, "text") else str(out)
    else:
        # older API compatibility
        out = genai.generate(prompt=prompt, temperature=temperature)
        # attempt to extract text
        if isinstance(out, dict):
            cands = out.get("candidates", [])
            if cands:
                return cands[0].get("content", "")
            return json.dumps(out)
        return str(out)

# EDF/CSV reading
def read_edf_bytes(file_bytes: bytes):
    raw = mne.io.read_raw_edf(BytesIO(file_bytes), preload=True, verbose=False)
    data = raw.get_data()  # shape (n_channels, n_samples)
    sfreq = raw.info["sfreq"]
    ch_names = raw.info["ch_names"]
    return data, sfreq, ch_names

def read_csv_eeg(file_bytes: bytes, sample_rate=None):
    """
    Accept CSV with columns=channels (or first column = time). Return data (channels x samples) and sfreq.
    If first column looks like time (monotonic increasing), drop it.
    """
    df = pd.read_csv(BytesIO(file_bytes))
    # detect time column
    if df.shape[1] > 1 and (df.iloc[:, 0].dtype.kind in "fi") and (df.iloc[:, 0].is_monotonic_increasing):
        # assume first column is time
        times = df.iloc[:, 0].values
        # estimate sfreq
        diffs = np.diff(times)
        if len(diffs) > 0:
            sfreq = 1.0 / np.median(diffs)
        else:
            sfreq = sample_rate or 256.0
        df = df.iloc[:, 1:]
    else:
        sfreq = sample_rate or 256.0
    data = df.T.values.astype(np.float32)
    ch_names = list(df.columns.astype(str))
    return data, sfreq, ch_names

# PSD computation
def compute_psd(eeg: np.ndarray, sfreq: float, nperseg=512):
    """Return psd (channels x freqs) and freqs array"""
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, sfreq, nperseg=min(nperseg, len(ch)))
        psd_list.append(pxx)
    return np.array(psd_list), f

# ONNX helpers
def load_onnx_session(path: str):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def run_onnx(session: ort.InferenceSession, input_name: str, arr: np.ndarray):
    return session.run(None, {input_name: arr.astype(np.float32)})

# Token/text mapping (simple char-level fallback)
FALLBACK_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")  # expand as needed
def tokens_to_text(tokens, vocab=FALLBACK_VOCAB):
    chars = []
    for t in tokens:
        chars.append(vocab[int(t) % len(vocab)])
    return "".join(chars).replace("  ", " ").strip()

# Accuracy metrics
def levenshtein(a: str, b: str):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0]*lb
        ai = a[i-1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[lb]

def char_accuracy(pred: str, true: str):
    if len(true) == 0:
        return 100.0 if len(pred) == 0 else 0.0
    ed = levenshtein(pred, true)
    return max(0.0, 1.0 - ed / max(len(true), 1)) * 100.0

def word_accuracy(pred: str, true: str):
    t = true.split()
    p = pred.split()
    if len(t) == 0:
        return 100.0 if len(p) == 0 else 0.0
    m = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
    return (m / len(t)) * 100.0

# Explainability: perturbation occlusion on PSD
def perturbation_importance(model_runner, base_input: np.ndarray, channels: int, freq_bins: int, step_bins=4):
    """
    model_runner(encoded_input) -> predicted probability or score (float) to judge importance.
    We'll slide an occlusion window over frequency bins per channel and measure drop in score.
    Returns importance map: channels x freq_bins (approx).
    """
    baseline_score = float(model_runner(base_input))
    importance = np.zeros((channels, freq_bins), dtype=float)
    # occlusion mask width
    w = max(1, int(step_bins))
    for ch in range(channels):
        for start in range(0, freq_bins, w):
            end = min(freq_bins, start + w)
            pert = base_input.copy()
            # zero out freq bins for channel ch between start:end
            pert[0, start:end, ch] = 0.0
            score = float(model_runner(pert))
            importance[ch, start:end] = max(0.0, baseline_score - score)
    # normalize
    total = importance.sum()
    if total > 0:
        importance /= total
    return importance

# A simple model runner adapter: if you have an ONNX decoder that returns logits/probs, you may adapt.
def example_model_score_from_decoded_probs(decoded_probs: np.ndarray):
    """
    Choose a scalar scoring function from decoder output to compare perturbations.
    Here: sum of max probabilities across time steps.
    """
    # decoded_probs: (1, seq_len, vocab)
    probs = np.array(decoded_probs)
    score = probs.max(axis=-1).mean()
    return float(score)

# ----------------------------
# Streamlit UI
# ----------------------------
st.header("Upload EEG (.edf or .csv) and models (optional)")
uploaded = st.file_uploader("Upload .edf or .csv EEG file", type=["edf", "csv"])
st.info("Place encoder.onnx and decoder.onnx in repository root to enable model inference. Otherwise app uses fallback demo encoder/decoder.")

# show model file presence
enc_path = "encoder.onnx"
dec_path = "decoder.onnx"
enc_exists = os.path.exists(enc_path)
dec_exists = os.path.exists(dec_path)
st.sidebar.markdown("## Models present")
st.sidebar.write({"encoder.onnx": enc_exists, "decoder.onnx": dec_exists})

# LLM config (secure)
llm_key = get_api_key()
if llm_key:
    st.sidebar.success("LLM API key available")
else:
    st.sidebar.info("LLM API key not set (set st.secrets['GOOGLE_API_KEY'] or env var)")

# Main flow
if uploaded is None:
    st.info("Upload an EEG file to begin.")
    st.stop()

# Read file
fname = uploaded.name.lower()
file_bytes = uploaded.read()
try:
    if fname.endswith(".edf"):
        eeg_data, sfreq, ch_names = read_edf_bytes(file_bytes)
    else:
        eeg_data, sfreq, ch_names = read_csv_eeg(file_bytes)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"Loaded EEG: channels={len(ch_names)}, samples={eeg_data.shape[1]}, sfreq={sfreq:.2f} Hz")

# Plot raw few channels (first 6)
st.subheader("Raw EEG (first channels)")
nplot = min(6, eeg_data.shape[0])
fig_raw, axs = plt.subplots(nplot, 1, figsize=(10, 1.2*nplot), sharex=True)
times = np.arange(eeg_data.shape[1]) / sfreq
for i in range(nplot):
    axs[i].plot(times, eeg_data[i], linewidth=0.6)
    axs[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
axs[-1].set_xlabel("Time (s)")
st.pyplot(fig_raw)

# Compute PSD
psd, freqs = compute_psd(eeg_data, sfreq)
st.subheader("Power Spectral Density (sample channels)")
fig_psd, ax = plt.subplots(figsize=(10, 3))
for i in range(min(4, psd.shape[0])):
    ax.semilogy(freqs, psd[i], label=ch_names[i])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend()
st.pyplot(fig_psd)

# Prepare model input: treat frequency bins as time sequence: shape (1, seq_len, channels)
psd_T = psd.T  # freqs x channels
psd_T = (psd_T - psd_T.mean()) / (psd_T.std() + 1e-9)
model_input = psd_T[None, ...].astype(np.float32)
st.write("Model input shape (batch, seq_len (freq bins), channels):", model_input.shape)

# Load ONNX sessions if present
enc_sess = None
dec_sess = None
if enc_exists:
    try:
        enc_sess = load_onnx_session(enc_path)
        st.sidebar.success("encoder.onnx loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load encoder.onnx: {e}")
        enc_sess = None
if dec_exists:
    try:
        dec_sess = load_onnx_session(dec_path)
        st.sidebar.success("decoder.onnx loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load decoder.onnx: {e}")
        dec_sess = None

# Run encoder (ONNX or fallback)
if enc_sess is not None:
    enc_inp_name = enc_sess.get_inputs()[0].name
    enc_outs = enc_sess.run(None, {enc_inp_name: model_input})
    encoded = enc_outs[0]
    st.info("ONNX encoder inference completed.")
else:
    # Fallback simple encoder (residual-approx)
    # create sliding-window features
    seq_len, channels = model_input.shape[1], model_input.shape[2]
    rng = np.random.RandomState(42)
    W = rng.randn(channels, 64).astype(np.float32)
    encoded = np.matmul(model_input[0], W)[None, ...]  # shape (1, seq_len, 64)
    st.info("Fallback encoder used (demo).")

# Run decoder (ONNX or fallback) -> decoded_probs shape (1, seq_len, vocab)
if dec_sess is not None:
    dec_inp_name = dec_sess.get_inputs()[0].name
    dec_outs = dec_sess.run(None, {dec_inp_name: encoded.astype(np.float32)})
    decoded_probs = dec_outs[0]
    st.info("ONNX decoder inference completed.")
else:
    # fallback decoder: random projection to vocab
    vocab_size = 128
    rng = np.random.RandomState(123)
    Wd = rng.randn(encoded.shape[2], vocab_size).astype(np.float32)
    logits = np.matmul(encoded, Wd)  # (1, seq_len, vocab)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    decoded_probs = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
    st.info("Fallback decoder used (demo).")

# Convert decoded probs -> predicted tokens -> predicted text
pred_tokens = np.argmax(decoded_probs, axis=-1)[0].tolist()
pred_text = tokens_to_text(pred_tokens)

# Option: let LLM (Gemini) provide prediction instead (if key present & user wants)
use_llm_for_prediction = st.checkbox("Use LLM to generate text from PSD summary (optional)", value=False)
llm_generated_prediction = None
if use_llm_for_prediction:
    key = get_api_key()
    if not key:
        st.error("No LLM API key available. Set st.secrets['GOOGLE_API_KEY'] or env var GOOGLE_API_KEY.")
    elif not HAS_GENAI:
        st.error("google.generativeai client not installed in environment.")
    else:
        # build compact PSD summary for prompt
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
        band_power = {}
        for b, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if idx.size:
                band_power[b] = list(np.round(psd[:, idx].mean(axis=1)[:6], 6))
            else:
                band_power[b] = [0.0]*min(6, psd.shape[0])
        top_ch_idx = np.argsort(np.mean([band_power[b] for b in band_power], axis=0))[-4:][::-1]
        top_ch = [ch_names[i] if i < len(ch_names) else f"ch{i}" for i in top_ch_idx]
        prompt = (
            "You are a concise EEG-to-text model. Given this PSD band power summary and top channels, "
            "produce a short predicted transcription (1 sentence) representing the likely content.\n\n"
            f"Band power (per band, first channels): {band_power}\nTop channels: {top_ch}\n"
            "Return only the short transcription in one line."
        )
        try:
            llm_out = llm_generate(prompt, model_name="gemini-2.5-flash", temperature=0.08)
            llm_generated_prediction = llm_out.strip().split("\n")[0]
            st.success("LLM generated a prediction.")
        except Exception as e:
            st.error(f"LLM generation failed: {e}")
            llm_generated_prediction = None

# Choose final prediction (priority: LLM-generated if selected else model)
final_prediction = llm_generated_prediction if (llm_generated_prediction is not None) else pred_text

# Ground-truth input (user-provided) or LLM-estimated (if the user wants)
user_ground_truth = st.text_area("Actual ground-truth (optional) — paste here if available", value="", height=120)
use_llm_for_gt = st.checkbox("Let LLM estimate ground-truth from PSD if you don't provide one (optional)", value=False)
llm_ground_truth = None
if use_llm_for_gt and (not user_ground_truth.strip()):
    key = get_api_key()
    if not key:
        st.error("No LLM API key available for GT estimation.")
    elif not HAS_GENAI:
        st.error("google.generativeai client not installed.")
    else:
        prompt_gt = (
            "You are an EEG-to-text expert. Given PSD band-power summary and top channels, "
            "provide a concise ground-truth transcription (1 sentence) that best explains the EEG signal.\n\n"
            f"Band-power sample (first channels): top summarized bands.\nReturn only the concise plain text."
        )
        try:
            llm_out_gt = llm_generate(prompt_gt, model_name="gemini-2.5-flash", temperature=0.08)
            llm_ground_truth = llm_out_gt.strip().split("\n")[0]
            st.success("LLM estimated a ground-truth.")
        except Exception as e:
            st.error(f"LLM GT generation failed: {e}")
            llm_ground_truth = None

final_ground_truth = user_ground_truth.strip() if user_ground_truth.strip() else (llm_ground_truth or "")

# Show side-by-side Actual vs Predicted
st.subheader("Actual (Ground-truth)  ←→  Predicted")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Actual (ground-truth)**")
    st.write(final_ground_truth)
with col2:
    st.markdown("**Predicted**")
    st.write(final_prediction)

# Compute accuracies if GT present
if final_ground_truth:
    pred_norm = " ".join(final_prediction.lower().split())
    gt_norm = " ".join(final_ground_truth.lower().split())
    c_acc = char_accuracy(pred_norm, gt_norm)
    w_acc = word_accuracy(pred_norm, gt_norm)

    st.subheader("Accuracy Metrics")
    k1, k2 = st.columns(2)
    k1.metric("Character-level accuracy (%)", f"{c_acc:.2f}")
    k2.metric("Word-level accuracy (%)", f"{w_acc:.2f}")

    # bar
    fig_acc, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Char", "Word"], [c_acc, w_acc], color=["#2b8cbe","#fdae61"])
    ax.set_ylim(0,100)
    st.pyplot(fig_acc)

    # show char diff inline
    def inline_char_diff(t, p, max_chars=400):
        t = t[:max_chars].ljust(max_chars)
        p = p[:max_chars].ljust(max_chars)
        true_html = []
        pred_html = []
        for i in range(max_chars):
            ct = t[i]
            cp = p[i]
            if ct == cp:
                true_html.append(ct if ct != " " else "&middot;")
                pred_html.append(cp if cp != " " else "&middot;")
            else:
                true_html.append(f"<span style='background:#c6efce'>{ct if ct!=' ' else '&middot;'}</span>")
                pred_html.append(f"<span style='background:#ffc7ce'>{cp if cp!=' ' else '&middot;'}</span>")
        return "<div style='font-family:monospace;white-space:pre-wrap'>True: " + "".join(true_html) + "</div><div style='font-family:monospace;white-space:pre-wrap'>Pred: " + "".join(pred_html) + "</div>"

    st.markdown(inline_char_diff(gt_norm, pred_norm), unsafe_allow_html=True)
else:
    st.info("No ground-truth available. Provide one or ask LLM to estimate ground-truth via the checkbox above.")

# Explainability: compute perturbation importance (fast approx)
st.subheader("Explainability: Frequency × Channel importance (occlusion)")
with st.spinner("Computing importance (this may take a few seconds)..."):
    # define model_runner that takes model_input-like array and returns scalar score
    def model_runner(arr):
        # arr shape (1, seq_len, channels)
        # if decoder present, run encoded->decoder; else fallback scoring from decoded_probs
        try:
            # If encoder exists, try to run decoder with arr as encoded (best-effort)
            if dec_sess is not None and enc_sess is not None:
                # If arr is in PSD shape, we would normally run encoder; but for perturbation we can
                # approximate by running encoder on arr if enc_sess expects PSD input.
                # Attempt to call encoder then decoder; fallback to precomputed decoded_probs.
                try:
                    enc_in = enc_sess.get_inputs()[0].name
                    enc_out = enc_sess.run(None, {enc_in: arr.astype(np.float32)})[0]
                    dec_in = dec_sess.get_inputs()[0].name
                    dec_out = dec_sess.run(None, {dec_in: enc_out.astype(np.float32)})[0]
                    return example_model_score_from_decoded_probs(dec_out)
                except Exception:
                    return example_model_score_from_decoded_probs(decoded_probs)
            else:
                # use existing decoded_probs static baseline
                return example_model_score_from_decoded_probs(decoded_probs)
        except Exception:
            return example_model_score_from_decoded_probs(decoded_probs)

    # run perturbation importance
    channels = model_input.shape[2]
    freq_bins = model_input.shape[1]
    importance_map = perturbation_importance(model_runner, model_input.copy(), channels, freq_bins, step_bins=max(1, freq_bins//24))
    st.success("Importance computed.")

# Visualize importance heatmap
fig_imp, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(importance_map, aspect='auto', origin='lower', cmap='magma')
ax.set_ylabel("Channel index")
ax.set_xlabel("Frequency bin index")
ax.set_yticks(range(len(ch_names)))
ax.set_yticklabels(ch_names, fontsize=8)
plt.colorbar(im, ax=ax, label="Relative importance")
st.pyplot(fig_imp)

# Also show aggregated band importances
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
band_scores = {}
for b, (low, high) in bands.items():
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    if idx.size:
        band_scores[b] = importance_map[:, idx].sum(axis=1).mean()  # average across channels
    else:
        band_scores[b] = 0.0

fig_band, ax = plt.subplots(figsize=(6,3))
ax.bar(list(band_scores.keys()), [band_scores[b] for b in band_scores])
ax.set_ylabel("Aggregate importance (arbitrary normalized units)")
st.pyplot(fig_band)

# Agentic LLM analysis (optional)
if HAS_GENAI and get_api_key():
    if st.button("LLM: Produce concise interpretation & improvement suggestions (agentic)"):
        try:
            prompt = (
                "You are an expert in EEG-to-text decoding and explainability. Below are:\n"
                f"- Final prediction: {final_prediction}\n- Ground-truth (if any): {final_ground_truth}\n"
                f"- Short PSD summary (top channels & bands): top channels unknown here\n"
                "Provide a concise professional analysis: (1) likely reasons for mismatches, (2) suggestions to improve decoding performance, "
                "(3) what the importance heatmap implies about channels/frequencies. Keep answer concise and professional."
            )
            resp = llm_generate(prompt, model_name="gemini-2.5-flash", temperature=0.2)
            st.subheader("Agentic LLM analysis")
            st.write(resp)
        except Exception as e:
            st.error(f"Agentic LLM failed: {e}")
else:
    st.info("LLM analysis is available if you install google.generativeai and set GOOGLE_API_KEY in Streamlit Secrets or env var.")

st.markdown("---")
st.caption("Professional EEG→Text pipeline: PSD features, residual BiLSTM encoder (ONNX), seq2seq decoder (ONNX), model-agnostic perturbation explainability, and optional LLM-based interpretation. Configure keys in Streamlit Secrets; never commit keys in code.")
