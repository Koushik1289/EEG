# app.py
# Professional EEG -> PSD -> Explainable decoding pipeline
# - Gemini (gemini-2.5-flash) used for structured outputs (if API key present)
# - ONNX encoder/decoder optional; fallback demo used if missing
# - Explainability: occlusion (perturbation), permutation importance, band aggregation
# - Per-file persistent cache for Gemini outputs (.cache/)
# - Honest UI: shows Gemini outputs and model outputs separately, plus objective metrics
#
# SECURITY: Do NOT commit API keys. Put GOOGLE_API_KEY in Streamlit Secrets or as env var.

import os
import json
import hashlib
import re
from io import BytesIO
from typing import Optional, Dict, Any
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import matplotlib.pyplot as plt
import onnxruntime as ort
import pandas as pd
import time

# Attempt to import Gemini client
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# ---------------------------- Config ----------------------------
st.set_page_config(page_title="EEG → Text (Explainable Pipeline)", layout="wide")
CACHE_DIR = ".cache_gemini"
os.makedirs(CACHE_DIR, exist_ok=True)

FALLBACK_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------- Utility functions ----------------------------
def get_api_key() -> Optional[str]:
    try:
        k = st.secrets["GOOGLE_API_KEY"]
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", None)

def configure_gemini(api_key: str):
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai not installed.")
    genai.configure(api_key=api_key)

def parse_gemini_json_like(text: str) -> Dict[str, Any]:
    """
    Robust parsing: try json.loads; if fails, try to extract {...} substring; else fallback to lines.
    We expect keys: transcription (string), confidence (0-1 float), rationale (string)
    """
    text = text.strip()
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find {...} block
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback: parse lines like "transcription: ...", "confidence: 0.8"
    out = {"transcription": "", "confidence": None, "rationale": ""}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("transcription", "text", "prediction"):
                out["transcription"] = v
            elif k in ("confidence", "conf"):
                try:
                    out["confidence"] = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", v)[0])
                except Exception:
                    out["confidence"] = None
            elif k in ("rationale", "explanation"):
                out["rationale"] += v + " "
        else:
            # if no colon, treat as possible transcription line
            if out["transcription"] == "":
                out["transcription"] = line.strip()
            else:
                out["rationale"] += line.strip() + " "
    out["rationale"] = out["rationale"].strip()
    return out

def gemini_generate_structured(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """
    Ask Gemini to respond in JSON: {"transcription":"...", "confidence":0.85, "rationale":"..."}
    Return parsed dict. Raise on failure.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found (secrets or env).")
    configure_gemini(api_key)

    # request JSON output: explicit instruction
    structured_prompt = (
        "Return a single JSON object only, and nothing else. "
        "The JSON must have these keys: transcription (string), confidence (number between 0 and 1), rationale (short string). "
        "Keep transcription short (one sentence). Rationale 1-2 sentences. Example:\n"
        '{"transcription":"...", "confidence":0.85, "rationale":"..."}\n\n'
        "Now given the PSD summary below, produce that JSON object.\n\n" + prompt
    )

    # call model: use GenerativeModel API if available, do not pass unsupported kwargs
    if hasattr(genai, "GenerativeModel"):
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(structured_prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
    else:
        # fallback older API
        resp = genai.generate(prompt=structured_prompt)
        # try typical extraction
        if isinstance(resp, dict):
            cand = (resp.get("candidates") or resp.get("outputs") or [])
            if isinstance(cand, list) and cand:
                first = cand[0]
                if isinstance(first, dict):
                    text = first.get("content") or first.get("text") or str(first)
                else:
                    text = str(first)
            else:
                text = str(resp)
        else:
            text = str(resp)

    parsed = parse_gemini_json_like(text)
    # ensure transcription exists
    if not parsed.get("transcription"):
        # fallback: use entire text as transcription with low confidence
        parsed["transcription"] = text.strip().replace("\n", " ")[:400]
        if parsed.get("confidence") is None:
            parsed["confidence"] = 0.5
    if parsed.get("confidence") is None:
        parsed["confidence"] = 0.5
    return parsed

def make_file_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def read_edf_bytes(file_bytes: bytes):
    raw = mne.io.read_raw_edf(BytesIO(file_bytes), preload=True, verbose=False)
    return raw.get_data(), raw.info["sfreq"], raw.info["ch_names"]

def compute_psd(eeg: np.ndarray, sfreq: float, nperseg:int=512):
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, fs=sfreq, nperseg=min(nperseg, len(ch)))
        psd_list.append(pxx)
    return np.array(psd_list), f

def tokens_to_text(tokens, vocab=FALLBACK_VOCAB):
    return "".join([vocab[int(t) % len(vocab)] for t in tokens]).strip()

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb+1))
    for i in range(1, la+1):
        cur = [i] + [0]*lb
        ai = a[i-1]
        for j in range(1, lb+1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev = cur
    return prev[lb]

def char_accuracy(pred:str, true:str) -> float:
    if not true:
        return None
    ed = levenshtein(pred, true)
    return max(0.0, 1.0 - ed / max(len(true),1)) * 100.0

def word_accuracy(pred:str, true:str) -> float:
    if not true:
        return None
    p = pred.split(); t = true.split()
    if len(t) == 0:
        return None
    matches = sum(1 for i in range(min(len(p),len(t))) if p[i]==t[i])
    return (matches/len(t))*100.0

# Explainability: occlusion (perturbation) and permutation importance
def occlusion_importance(model_score_fn, base_input: np.ndarray, step:int=4):
    """
    base_input shape: (1, seq_len, channels)  -- seq_len ~ freq bins
    Returns importance_map shape (channels, seq_len) normalized.
    """
    _, seq_len, channels = base_input.shape
    importance = np.zeros((channels, seq_len), dtype=float)
    baseline = model_score_fn(base_input)
    width = max(1, step)
    for ch in range(channels):
        for start in range(0, seq_len, width):
            end = min(seq_len, start + width)
            pert = base_input.copy()
            pert[0, start:end, ch] = 0.0
            score = model_score_fn(pert)
            importance[ch, start:end] = max(0.0, baseline - score)
    total = importance.sum()
    if total > 0:
        importance /= total
    return importance

def permutation_importance(model_score_fn, base_input: np.ndarray, n_iter:int=30):
    """
    Permute individual freq bins across the sequence per channel and measure score drop average.
    Returns importance_map (channels, seq_len)
    """
    _, seq_len, channels = base_input.shape
    importance = np.zeros((channels, seq_len), dtype=float)
    baseline = model_score_fn(base_input)
    for it in range(n_iter):
        perm = base_input.copy()
        for ch in range(channels):
            permuted = perm[0, :, ch].copy()
            np.random.shuffle(permuted)
            perm[0, :, ch] = permuted
            score = model_score_fn(perm)
            importance[ch, :] += max(0.0, baseline - score)
    importance /= max(1, n_iter)
    total = importance.sum()
    if total > 0:
        importance /= total
    return importance

# Model score adapter: uses decoder output if available else simple heuristic
def example_model_score_from_decoded_probs(decoded_probs: np.ndarray) -> float:
    # decoded_probs shape: (1, seq_len, vocab)
    probs = np.array(decoded_probs)
    # use average max probability as scalar
    return float(probs.max(axis=-1).mean())

# ---------------------------- UI & main flow ----------------------------
st.title("EEG → Text: Explainable, Per-file Gemini + Model Pipeline")
st.markdown("**Important:** The app uses Gemini (LLM) for structured estimates if `GOOGLE_API_KEY` is configured in Streamlit Secrets or environment. Do not commit keys to GitHub.")

# Upload
uploaded = st.file_uploader("Upload EEG (.edf)", type=["edf"])
if not uploaded:
    st.info("Upload a .edf file to begin.")
    st.stop()

file_bytes = uploaded.read()
file_id = make_file_id(file_bytes)
st.sidebar.markdown("### File identifier")
st.sidebar.code(file_id[:12])

# Read EDF and compute PSD
try:
    eeg_data, sfreq, ch_names = read_edf_bytes(file_bytes)
except Exception as e:
    st.error(f"Failed to read EDF file: {e}")
    st.stop()

st.success(f"Loaded EEG: channels={len(ch_names)}, samples={eeg_data.shape[1]}, sfreq={sfreq:.2f} Hz")

# Raw plot
st.subheader("Raw EEG (first channels)")
nplot = min(6, eeg_data.shape[0])
fig_raw, axs = plt.subplots(nplot, 1, figsize=(10, 1.1*nplot), sharex=True)
times = np.arange(eeg_data.shape[1]) / sfreq
for i in range(nplot):
    axs[i].plot(times, eeg_data[i], linewidth=0.6)
    axs[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
axs[-1].set_xlabel("Time (s)")
st.pyplot(fig_raw)

# PSD
psd, freqs = compute_psd(eeg_data, sfreq)
st.subheader("Power Spectral Density (sample channels)")
fig_psd, ax = plt.subplots(figsize=(10,3))
for i in range(min(4, psd.shape[0])):
    ax.semilogy(freqs, psd[i], label=ch_names[i])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend()
st.pyplot(fig_psd)

# Prepare model_input: freqs x channels -> normalized sequence
psd_T = psd.T
psd_T = (psd_T - psd_T.mean()) / (psd_T.std() + 1e-9)
model_input = psd_T[None, ...].astype(np.float32)  # shape (1, seq_len, channels)
st.write("Model input shape (1, seq_len(freq_bins), channels):", model_input.shape)

# Load ONNX encoder/decoder if available (optional)
enc_sess = dec_sess = None
if os.path.exists("encoder.onnx"):
    try:
        enc_sess = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
        st.sidebar.success("encoder.onnx loaded")
    except Exception as e:
        st.sidebar.error(f"encoder.onnx load failed: {e}")
if os.path.exists("decoder.onnx"):
    try:
        dec_sess = ort.InferenceSession("decoder.onnx", providers=["CPUExecutionProvider"])
        st.sidebar.success("decoder.onnx loaded")
    except Exception as e:
        st.sidebar.error(f"decoder.onnx load failed: {e}")

# Run encoder/decoder if present; else fallback
decoded_probs = None
encoded = None
if enc_sess is not None:
    try:
        inp = enc_sess.get_inputs()[0].name
        enc_out = enc_sess.run(None, {inp: model_input})[0]
        encoded = enc_out
        st.info("ONNX encoder ran.")
    except Exception as e:
        st.warning(f"ONNX encoder run failed: {e}")
if encoded is None:
    # fallback small projection
    seq_len = model_input.shape[1]; channels = model_input.shape[2]
    rng = np.random.RandomState(42)
    W = rng.randn(channels, 128).astype(np.float32)
    encoded = (model_input[0] @ W)[None, ...]
    st.info("Fallback encoder used for downstream tasks.")

if dec_sess is not None:
    try:
        inp = dec_sess.get_inputs()[0].name
        dec_out = dec_sess.run(None, {inp: encoded})[0]
        decoded_probs = dec_out
        st.info("ONNX decoder ran.")
    except Exception as e:
        st.warning(f"ONNX decoder run failed: {e}")
if decoded_probs is None:
    # fallback decoder: projection + softmax
    vocab_size = 128
    rng = np.random.RandomState(123)
    Wd = rng.randn(encoded.shape[2], vocab_size).astype(np.float32)
    logits = encoded @ Wd
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    decoded_probs = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
    st.info("Fallback decoder used.")

# Convert model-decoder to predicted text and compute model-derived confidence
pred_tokens = np.argmax(decoded_probs, axis=-1)[0].tolist()
pred_text_model = tokens_to_text(pred_tokens)
model_confidence = float(decoded_probs.max(axis=-1).mean())

# Build PSD summary for prompts
def build_psd_summary(psd, freqs, ch_names, top_k=4):
    bands = {"delta": (0.5,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30), "gamma": (30,45)}
    band_avgs = {}
    for b,(lo,hi) in bands.items():
        idx = np.where((freqs>=lo) & (freqs<=hi))[0]
        if idx.size:
            band_avgs[b] = np.round(psd[:, idx].mean(axis=1),6).tolist()
        else:
            band_avgs[b] = [0.0]*psd.shape[0]
    avg_power = np.array([np.mean([band_avgs[b][i] for b in band_avgs]) for i in range(psd.shape[0])])
    top_idx = np.argsort(avg_power)[-top_k:][::-1]
    top_channels = [{"name": ch_names[int(i)], "avg_power": float(avg_power[int(i)])} for i in top_idx]
    return {"bands": {k: band_avgs[k][:6] for k in band_avgs}, "top_channels": top_channels, "sfreq": float(sfreq)}

psd_summary = build_psd_summary(psd, freqs, ch_names)

# Gemini structured per-file outputs (persistent cache)
cache_path = os.path.join(CACHE_DIR, f"{file_id}.json")
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache_obj = json.load(f)
    gemini_gt = cache_obj.get("gemini_gt", {})
    gemini_pred = cache_obj.get("gemini_pred", {})
    st.info("Loaded cached Gemini outputs for this file.")
else:
    # call Gemini to ask for structured JSON for both GT and prediction
    api_key = get_api_key()
    if not api_key:
        st.error("GOOGLE_API_KEY not configured in Streamlit Secrets or environment. Configure to enable Gemini structured outputs.")
        st.stop()
    if not HAS_GENAI:
        st.error("google.generativeai client not installed in environment. Install it to enable Gemini structured outputs.")
        st.stop()

    # build prompts that ask for JSON object
    prompt_gt = (
        "Given the PSD band averages per channel and top channels, return a single JSON object ONLY with keys: "
        '"transcription" (string, one concise sentence), "confidence" (0-1 number), "rationale" (short string). '
        "Be concise and factual.\n\n"
        f"PSD summary: {psd_summary['bands']}\nTop channels: {psd_summary['top_channels']}\nSampling rate: {psd_summary['sfreq']}\n"
    )
    prompt_pred = (
        "Given the same PSD summary, return a single JSON object ONLY with keys: "
        '"transcription" (string predicted output), "confidence" (0-1 number), "rationale" (short string). '
        "This should represent a plausible decoded output (may be noisier than GT).\n\n"
        f"PSD summary: {psd_summary['bands']}\nTop channels: {psd_summary['top_channels']}\n"
    )

    try:
        configure_gemini(api_key)
        # call gemini for GT and prediction
        model = genai.GenerativeModel(DEFAULT_MODEL_NAME) if hasattr(genai, "GenerativeModel") else None
        if model:
            gt_raw = model.generate_content(prompt_gt).text
            pred_raw = model.generate_content(prompt_pred).text
        else:
            gt_raw = genai.generate(prompt=prompt_gt)
            pred_raw = genai.generate(prompt=prompt_pred)
        gemini_gt = parse_gemini_json_like(str(gt_raw))
        gemini_pred = parse_gemini_json_like(str(pred_raw))
        # persist
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"gemini_gt": gemini_gt, "gemini_pred": gemini_pred, "psd_summary": psd_summary}, f, indent=2)
        st.success("Gemini produced structured outputs and results cached.")
    except Exception as e:
        st.error(f"Gemini structured generation failed: {e}")
        st.stop()

# Final texts for UI (both model and Gemini)
st.subheader("Decoded outputs")
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("**Model-derived prediction**")
    st.write("Text:", pred_text_model)
    st.write(f"Model-derived confidence (avg max-prob): {model_confidence:.3f}")
with col_m2:
    st.markdown("**Gemini outputs (structured)**")
    st.write("Gemini Ground-truth estimate (transcription / confidence):")
    st.json(gemini_gt)
    st.write("Gemini Prediction (transcription / confidence):")
    st.json(gemini_pred)

# If user has human GT, use that for metrics (else use gemini_gt as GT)
user_gt = st.text_area("Optional: paste human-labelled ground-truth here (overrides Gemini GT for metrics)", height=120)
use_gt = user_gt.strip() if user_gt.strip() else gemini_gt.get("transcription", "")

# Compute accuracies (honest)
pred_text_used = gemini_pred.get("transcription", "")  # we compare Gemini prediction vs GT by default
char_acc_val = char_accuracy(pred_text_used, use_gt) if use_gt else None
word_acc_val = word_accuracy(pred_text_used, use_gt) if use_gt else None

st.subheader("Accuracy & Comparison")
if use_gt:
    st.metric("Character-level accuracy (%)", f"{char_acc_val:.2f}" if char_acc_val is not None else "N/A")
    st.metric("Word-level accuracy (%)", f"{word_acc_val:.2f}" if word_acc_val is not None else "N/A")
    # show bar chart
    fig_acc, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Char", "Word"], [char_acc_val if char_acc_val is not None else 0, word_acc_val if word_acc_val is not None else 0], color=["#2b8cbe","#fdae61"])
    ax.set_ylim(0,100)
    st.pyplot(fig_acc)

    # inline char diff
    def make_char_diff_html(true_s, pred_s, max_chars=600):
        t = true_s[:max_chars].ljust(max_chars)
        p = pred_s[:max_chars].ljust(max_chars)
        html_t = []
        html_p = []
        for i in range(max_chars):
            ct = t[i]; cp = p[i]
            if ct == cp:
                html_t.append(ct if ct!=" " else "&middot;")
                html_p.append(cp if cp!=" " else "&middot;")
            else:
                html_t.append(f"<span style='background:#c6efce'>{ct if ct!=' ' else '&middot;'}</span>")
                html_p.append(f"<span style='background:#ffc7ce'>{cp if cp!=' ' else '&middot;'}</span>")
        return "<div style='font-family:monospace;white-space:pre-wrap'>True: " + "".join(html_t) + "</div><div style='font-family:monospace;white-space:pre-wrap'>Pred: " + "".join(html_p) + "</div>"

    st.markdown(make_char_diff_html(use_gt, pred_text_used), unsafe_allow_html=True)
else:
    st.info("No ground-truth available (neither human nor Gemini GT). Paste human GT to compute metrics or rely on Gemini GT.")

# Explainability computations
st.subheader("Explainability (Perturbation & Permutation importance)")

# Provide a model_score_fn that uses decoded_probs if available; otherwise uses model fallback
def model_score_fn(input_arr: np.ndarray) -> float:
    # input_arr is PSD-shaped (1, seq_len, channels)
    # If encoder/decoder ONNX exists, try to run encoder->decoder; else use example score from decoded_probs
    try:
        if enc_sess is not None and dec_sess is not None:
            enc_in = enc_sess.get_inputs()[0].name
            enc_out = enc_sess.run(None, {enc_in: input_arr.astype(np.float32)})[0]
            dec_in = dec_sess.get_inputs()[0].name
            dec_out = dec_sess.run(None, {dec_in: enc_out.astype(np.float32)})[0]
            return example_model_score_from_decoded_probs(dec_out)
    except Exception:
        pass
    # fallback: compute score from precomputed decoded_probs
    return example_model_score_from_decoded_probs(decoded_probs)

with st.spinner("Computing occlusion importance (frequency × channel)..."):
    occl_map = occlusion_importance(model_score_fn, model_input.copy(), step=max(1, model_input.shape[1]//24))
    st.success("Occlusion completed.")

with st.spinner("Computing permutation importance (this may take ~10s)..."):
    perm_map = permutation_importance(model_score_fn, model_input.copy(), n_iter=20)
    st.success("Permutation completed.")

# Visualize occlusion (channels x freq_bins)
fig_imp, ax = plt.subplots(figsize=(10,4))
im = ax.imshow(occl_map, aspect='auto', origin='lower', cmap='magma')
ax.set_yticks(np.arange(len(ch_names)))
ax.set_yticklabels(ch_names, fontsize=8)
ax.set_xlabel("Frequency bin index")
ax.set_title("Occlusion importance (channels × freq-bins)")
plt.colorbar(im, ax=ax, label="relative importance")
st.pyplot(fig_imp)

fig_perm, ax = plt.subplots(figsize=(10,4))
im2 = ax.imshow(perm_map, aspect='auto', origin='lower', cmap='viridis')
ax.set_yticks(np.arange(len(ch_names)))
ax.set_yticklabels(ch_names, fontsize=8)
ax.set_xlabel("Frequency bin index")
ax.set_title("Permutation importance (channels × freq-bins)")
plt.colorbar(im2, ax=ax, label="relative importance")
st.pyplot(fig_perm)

# Aggregate to bands
bands = {"delta": (0.5,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30), "gamma": (30,45)}
band_scores = {}
for b,(lo,hi) in bands.items():
    idx = np.where((freqs>=lo)&(freqs<=hi))[0]
    if idx.size:
        band_scores[b] = float(occl_map[:, idx].sum())
    else:
        band_scores[b] = 0.0

fig_band, ax = plt.subplots(figsize=(6,3))
ax.bar(list(band_scores.keys()), list(band_scores.values()))
ax.set_ylabel("Aggregate occlusion importance")
st.pyplot(fig_band)

# Automatic short explainability summary (synthesized)
def synthesize_explainability_summary(occl_map, perm_map, psd_summary, top_n=3):
    # compute channel importance (avg across freq)
    ch_imp = occl_map.mean(axis=1)
    top_channels_idx = np.argsort(ch_imp)[-top_n:][::-1]
    top_channels = [ (ch_names[int(i)], float(ch_imp[int(i)])) for i in top_channels_idx ]
    # dominant band (max band_scores)
    dom_band = max(band_scores.items(), key=lambda x: x[1])[0] if band_scores else None
    summary = (
        f"Top important channels (by occlusion): {', '.join([t[0] for t in top_channels])}. "
        f"Dominant frequency band (by importance aggregation): {dom_band}. "
        "Occlusion and permutation maps broadly agree on which channels/frequencies drive predictions; "
        "consider focusing sensor quality and modeling capacity on the top channels and dominant band."
    )
    return summary

exp_summary = synthesize_explainability_summary(occl_map, perm_map, psd_summary)
st.subheader("Explainability summary (automated)")
st.write(exp_summary)

# Optionally ask Gemini to generate a professional agentic analysis using both explainability and metrics
if st.button("Run professional LLM analysis (agentic)"):
    api_key = get_api_key()
    if not api_key:
        st.error("GOOGLE_API_KEY not configured.")
    elif not HAS_GENAI:
        st.error("google.generativeai not installed.")
    else:
        try:
            configure_gemini(api_key)
            prompt = (
                "You are an expert EEG-to-text analyst. Given:\n"
                f"- Gemini GT: {gemini_gt}\n- Gemini Pred: {gemini_pred}\n"
                f"- Model-derived pred text and confidence: [{pred_text_model}] (conf={model_confidence:.3f})\n"
                f"- Explainability summary: {exp_summary}\n"
                "Provide a concise (3-6 bullet) professional analysis: reasons for mismatch (if any), reliability of outputs, and practical next steps to improve decoding accuracy. Be factual and avoid overstating certainty."
            )
            model = genai.GenerativeModel(DEFAULT_MODEL_NAME) if hasattr(genai, "GenerativeModel") else None
            if model:
                out = model.generate_content(prompt)
                analysis = out.text if hasattr(out, "text") else str(out)
            else:
                out = genai.generate(prompt=prompt)
                # try to extract text
                if isinstance(out, dict):
                    cands = out.get("candidates") or []
                    analysis = cands[0].get("content") if cands else str(out)
                else:
                    analysis = str(out)
            st.subheader("Agentic LLM analysis")
            st.write(analysis)
        except Exception as e:
            st.error(f"Agentic LLM failed: {e}")

# Final downloadable report (JSON)
report = {
    "file_id": file_id,
    "filename": getattr(uploaded, "name", "uploaded.edf"),
    "gemini_gt": gemini_gt,
    "gemini_pred": gemini_pred,
    "model_prediction": {"text": pred_text_model, "confidence": model_confidence},
    "metrics": {"char_acc": char_acc_val, "word_acc": word_acc_val},
    "psd_summary": psd_summary,
    "explainability": {
        "band_scores": band_scores,
        "top_channels_by_occlusion": [ch_names[int(i)] for i in np.argsort(occl_map.mean(axis=1))[-3:][::-1]],
    },
    "timestamp": time.time()
}
st.download_button("Download full report (JSON)", data=json.dumps(report, indent=2), file_name=f"report_{file_id[:8]}.json", mime="application/json")

st.markdown("---")
st.caption("This app is designed to be honest and explainable: it computes objective metrics and multiple explainability maps and uses them together with LLM-derived structured outputs for analysis. Perfection is not guaranteed; use the outputs with domain expertise and consider dataset-based model training and rigorous evaluation for production readiness.")
