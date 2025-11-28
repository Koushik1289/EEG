# app.py
# EEG -> PSD -> Gemini (per-file unique GT & Prediction) -> Compare & visualise
# Uses gemini-2.5-flash model via google.generativeai client if available.
#
# REQUIRED:
#  - Install google-generativeai and set GOOGLE_API_KEY in Streamlit Secrets or env var.
#  - Install: streamlit, numpy, scipy, mne, matplotlib, google-generativeai, onnxruntime (optional), pandas
#
# SECURITY: do NOT hardcode API keys here. Use Streamlit secrets or environment variables.

import os
import hashlib
import uuid
from io import BytesIO
from typing import Optional, Tuple
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import matplotlib.pyplot as plt

# Attempt to import the Gemini client
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# --------------------
# Helpers
# --------------------
def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, then environment
    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", None)

def configure_gemini_or_raise():
    key = get_api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not found in Streamlit secrets or environment variables.")
    # genai.configure exists in client
    genai.configure(api_key=key)

def gemini_generate_single(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Generate a single plain-text response from Gemini.
    Avoid passing unsupported keyword args to generate_content().
    Try modern GenerativeModel API first; fall back to genai.generate().
    """
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai client not installed in environment.")
    configure_gemini_or_raise()

    # Try modern client
    try:
        if hasattr(genai, "GenerativeModel"):
            m = genai.GenerativeModel(model_name)
            # call with only prompt text (no temperature kwarg to avoid previous error)
            out = m.generate_content(prompt)
            # Some clients return object with .text or .output[0].content, handle robustly
            if hasattr(out, "text"):
                return out.text.strip()
            # fallback: try stringifying
            return str(out).strip()
        else:
            # older client interface
            res = genai.generate(prompt=prompt)
            # res may be dict with candidates; handle common shapes
            if isinstance(res, dict):
                cands = res.get("candidates") or res.get("outputs") or []
                if cands and isinstance(cands, list):
                    first = cands[0]
                    if isinstance(first, dict):
                        # common key names
                        return (first.get("content") or first.get("text") or "").strip()
                    return str(first).strip()
                # fallback to top-level content keys
                return (res.get("content") or res.get("text") or str(res)).strip()
            return str(res).strip()
    except TypeError as te:
        # In case generate_content signature is different, try genai.generate as a safe fallback
        try:
            res = genai.generate(prompt=prompt)
            if isinstance(res, dict):
                cands = res.get("candidates") or res.get("outputs") or []
                if cands and isinstance(cands, list):
                    first = cands[0]
                    if isinstance(first, dict):
                        return (first.get("content") or first.get("text") or "").strip()
                    return str(first).strip()
            return str(res).strip()
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}") from e

def read_edf_bytes(file_bytes: bytes) -> Tuple[np.ndarray, float, list]:
    raw = mne.io.read_raw_edf(BytesIO(file_bytes), preload=True, verbose=False)
    return raw.get_data(), raw.info["sfreq"], raw.info["ch_names"]

def compute_psd(eeg: np.ndarray, sfreq: float, nperseg: int = 512):
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, fs=sfreq, nperseg=min(nperseg, len(ch)))
        psd_list.append(pxx)
    return np.array(psd_list), f

def make_file_id(file_bytes: bytes) -> str:
    # deterministic unique id per file: use sha256 of bytes, then shorten
    h = hashlib.sha256(file_bytes).hexdigest()
    # also add uuid4 suffix to avoid collisions across versions if needed (optional)
    return h

# Simple text metrics
def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())

def levenshtein(a: str, b: str) -> int:
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

def char_accuracy(pred: str, true: str) -> float:
    if len(true) == 0:
        return 100.0 if len(pred) == 0 else 0.0
    ed = levenshtein(pred, true)
    return max(0.0, 1.0 - ed / max(len(true), 1)) * 100.0

def word_accuracy(pred: str, true: str) -> float:
    t = true.split()
    p = pred.split()
    if len(t) == 0:
        return 100.0 if len(p) == 0 else 0.0
    m = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
    return (m / len(t)) * 100.0

# --------------------
# App UI & flow
# --------------------
st.set_page_config(page_title="EEG → Text (Gemini-driven) — Professional", layout="wide")
st.title("EEG → Text pipeline (Gemini-driven per-file GT & Prediction)")

st.markdown(
    "- Upload a `.edf` file. The system will compute PSD, build a compact PSD summary, and call Gemini to produce:"
)
st.markdown("  1. A **ground-truth estimate** (Gemini's best guess for the true transcription).")
st.markdown("  2. A **prediction** (Gemini's decoding output).")
st.markdown("- Both outputs are associated uniquely with the uploaded file and displayed side-by-side with accuracy metrics.")

# require API key and client
api_key = get_api_key()
if not HAS_GENAI:
    st.error("google.generativeai client library is not installed in the environment. Install it to enable Gemini.")
    st.stop()
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it to Streamlit Secrets or set environment variable GOOGLE_API_KEY.")
    st.stop()

uploaded = st.file_uploader("Upload .edf file", type=["edf"], accept_multiple_files=False)
if uploaded is None:
    st.info("Upload an EDF file to start.")
    st.stop()

file_bytes = uploaded.read()
file_id = make_file_id(file_bytes)

st.sidebar.markdown("### File unique id")
st.sidebar.code(file_id)

# compute EEG & PSD
try:
    eeg_data, sfreq, ch_names = read_edf_bytes(file_bytes)
except Exception as e:
    st.error(f"Failed to read EDF: {e}")
    st.stop()

st.success(f"Loaded EEG — channels: {len(ch_names)}, samples: {eeg_data.shape[1]}, sfreq: {sfreq} Hz")

# plot raw
st.subheader("Raw EEG (first channels)")
nplot = min(6, eeg_data.shape[0])
fig_raw, axes = plt.subplots(nplot, 1, figsize=(10, 1.1 * nplot), sharex=True)
times = np.arange(eeg_data.shape[1]) / sfreq
for i in range(nplot):
    axes[i].plot(times, eeg_data[i], linewidth=0.6)
    axes[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
axes[-1].set_xlabel("Time (s)")
st.pyplot(fig_raw)

# PSD
st.subheader("Power Spectral Density (example channels)")
psd, freqs = compute_psd(eeg_data, sfreq)
fig_psd, ax = plt.subplots(figsize=(9, 3))
for i in range(min(4, psd.shape[0])):
    ax.semilogy(freqs, psd[i], label=ch_names[i])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend(fontsize="small")
st.pyplot(fig_psd)

# Build compact summary for prompt
def summarize_for_prompt(psd: np.ndarray, freqs: np.ndarray, ch_names: list, top_k: int = 4):
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
    band_avgs = {}
    for b, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if idx.size:
            band_avgs[b] = np.round(psd[:, idx].mean(axis=1), 6).tolist()
        else:
            band_avgs[b] = [0.0] * psd.shape[0]
    # average across bands -> top channels
    avg_power = np.array([np.mean([band_avgs[b][i] for b in band_avgs]) for i in range(psd.shape[0])])
    top_idx = np.argsort(avg_power)[-top_k:][::-1]
    top_channels = [{"name": ch_names[int(i)], "avg_power": float(avg_power[int(i)])} for i in top_idx]
    return {"bands": {k: band_avgs[k][:6] for k in band_avgs}, "top_channels": top_channels, "sfreq": float(sfreq)}

psd_summary = summarize_for_prompt(psd, freqs, ch_names)

# Use session_state cache to store per-file outputs to avoid repeated LLM calls
if "file_llm_cache" not in st.session_state:
    st.session_state["file_llm_cache"] = {}

cache = st.session_state["file_llm_cache"]

if file_id in cache:
    st.info("Using cached Gemini outputs for this file (unique id matched).")
    gemini_gt = cache[file_id].get("gt", "")
    gemini_pred = cache[file_id].get("pred", "")
else:
    # Build prompts
    prompt_gt = (
        "You are an expert EEG-to-text decoder. Given the short PSD summary and top channels below, produce a single-line, "
        "concise ground-truth transcription that best explains the EEG signal. Output plain text (one line) only.\n\n"
        f"PSD band averages (truncated per channel): {psd_summary['bands']}\n"
        f"Top channels (name and avg_power): {psd_summary['top_channels']}\n"
        f"Sampling rate: {psd_summary['sfreq']} Hz\n\n"
        "Provide the most likely ground-truth transcription (one short sentence)."
    )

    prompt_pred = (
        "You are an EEG-to-text decoder producing the model's predicted transcription (may be noisier). Given the same PSD summary "
        "and top channels, output a single-line predicted transcription (one short sentence). Output plain text only.\n\n"
        f"PSD band averages (truncated): {psd_summary['bands']}\n"
        f"Top channels: {psd_summary['top_channels']}\n\n"
        "Provide the predicted transcription (one short sentence)."
    )

    # Call Gemini for GT and Prediction
    try:
        st.info("Generating ground-truth estimate and prediction using Gemini (this may take a few seconds)...")
        gemini_gt = gemini_generate_single(prompt_gt, model_name="gemini-2.5-flash")
        gemini_pred = gemini_generate_single(prompt_pred, model_name="gemini-2.5-flash")
        # store in cache
        cache[file_id] = {"gt": gemini_gt, "pred": gemini_pred}
        st.success("Gemini generation completed and cached for this file.")
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        st.stop()

# Display both (Gemini GT and Prediction). If you also have a manual ground-truth, let user paste to override display/metric.
st.subheader("Ground-truth (Gemini-estimated)  ←→  Prediction (Gemini)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Ground-truth (Gemini estimate)**")
    st.text_area("Gemini Ground-truth", value=gemini_gt, height=140)
with col2:
    st.markdown("**Prediction (Gemini output)**")
    st.text_area("Gemini Prediction", value=gemini_pred, height=140)

# If user has a human-labelled truth, allow paste (this will be used for metrics if provided)
st.markdown("### If you have a human-labelled ground-truth for this file, paste it below (optional).")
user_gt = st.text_area("Manual ground-truth (optional) — overrides Gemini GT for metrics", value="", height=120)
use_gt_for_metrics = user_gt.strip() != ""

# choose texts for comparison
gt_for_metrics = user_gt.strip() if use_gt_for_metrics else gemini_gt
pred_for_metrics = gemini_pred

# Compute metrics
gt_norm = normalize_text(gt_for_metrics)
pred_norm = normalize_text(pred_for_metrics)

char_acc = char_accuracy(pred_norm, gt_norm) if gt_for_metrics.strip() else None
word_acc = word_accuracy(pred_norm, gt_norm) if gt_for_metrics.strip() else None

st.subheader("Accuracy & Comparison")
if gt_for_metrics.strip():
    c1, c2 = st.columns(2)
    c1.metric("Character-level accuracy (%)", f"{char_acc:.2f}")
    c2.metric("Word-level accuracy (%)", f"{word_acc:.2f}")

    # bar plot
    fig_acc, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Character", "Word"], [char_acc, word_acc], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, 100)
    for i, v in enumerate([char_acc, word_acc]):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    st.pyplot(fig_acc)

    # inline diff (character)
    st.subheader("Character-level diff (highlights mismatches)")
    def make_char_diff(true_s: str, pred_s: str, max_chars: int = 600) -> str:
        t = true_s[:max_chars].ljust(max_chars)
        p = pred_s[:max_chars].ljust(max_chars)
        html_true, html_pred = [], []
        for i in range(max_chars):
            ct = t[i]
            cp = p[i]
            if ct == cp:
                html_true.append(ct if ct != " " else "&middot;")
                html_pred.append(cp if cp != " " else "&middot;")
            else:
                html_true.append(f"<span style='background:#c6efce'>{ct if ct!=' ' else '&middot;'}</span>")
                html_pred.append(f"<span style='background:#ffc7ce'>{cp if cp!=' ' else '&middot;'}</span>")
        return ("<div style='font-family:monospace;white-space:pre-wrap'>True: " + "".join(html_true)
                + "</div><div style='font-family:monospace;white-space:pre-wrap'>Pred: " + "".join(html_pred) + "</div>")

    st.markdown(make_char_diff(gt_norm, pred_norm), unsafe_allow_html=True)

    # word-level table
    st.subheader("Word-level comparison (position | actual | predicted | match)")
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
else:
    st.info("No human-labelled ground-truth provided — metrics will appear if you paste a manual ground-truth above.")

# Provide 'per-file' report download (JSON)
report = {
    "file_id": file_id,
    "filename": uploaded.name,
    "gemini_ground_truth": gemini_gt,
    "gemini_prediction": gemini_pred,
    "manual_ground_truth": user_gt.strip(),
    "char_accuracy": float(char_acc) if char_acc is not None else None,
    "word_accuracy": float(word_acc) if word_acc is not None else None,
    "psd_summary": psd_summary
}
st.download_button("Download report (JSON)", data=str(report), file_name=f"report_{file_id[:8]}.json", mime="application/json")

st.markdown("---")
st.caption("This system uses Gemini (LLM) to estimate both ground-truth and predictions per-file. Results are computed honestly; perfection is not guaranteed — the app displays the model outputs and objective metrics for analysis.")
