# app.py
# Streamlit EEG -> Text & Speech pipeline (transparent Gemini usage + explainability + postproc)
# REQUIRED: set GOOGLE_API_KEY in Streamlit Secrets or environment
# Optional: place encoder.onnx and decoder.onnx in repo root to enable ONNX inference
#
# Security: DO NOT hardcode API keys in this file.

import os, json, hashlib, time
from io import BytesIO
from typing import Optional, Dict, Any, List
import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import matplotlib.pyplot as plt
import onnxruntime as ort
from gtts import gTTS
import tempfile

# Gemini client (optional)
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

st.set_page_config(page_title="EEG→Text (Explainable)", layout="wide")
CACHE_DIR = ".cache_gemini"
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")

# ----------------- Helpers -----------------
def get_api_key() -> Optional[str]:
    try:
        k = st.secrets["GOOGLE_API_KEY"]
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", None)

def configure_gemini_or_raise():
    key = get_api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not found in Streamlit secrets or environment.")
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai client not installed.")
    genai.configure(api_key=key)

def robust_parse_json_like(text: str) -> Dict[str,Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find {...}
    import re
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback minimal parse
    return {"transcription": text.replace("\n"," ").strip(), "confidence": None, "rationale": ""}

def gemini_structured(prompt: str, model_name: str = DEFAULT_GEMINI_MODEL) -> Dict[str,Any]:
    """
    Request a single JSON object: transcription, confidence, rationale.
    """
    configure_gemini_or_raise()
    p = (
        "Return a single JSON object only with keys: "
        '"transcription" (string), "confidence" (0-1 number), "rationale" (short string). '
        "Example: {\"transcription\":\"...\",\"confidence\":0.85,\"rationale\":\"...\"}\n\n"
        + prompt
    )
    if hasattr(genai, "GenerativeModel"):
        model = genai.GenerativeModel(model_name)
        out = model.generate_content(p)
        text = out.text if hasattr(out, "text") else str(out)
    else:
        out = genai.generate(prompt=p)
        if isinstance(out, dict):
            cands = out.get("candidates") or out.get("outputs") or []
            if cands and isinstance(cands, list):
                first = cands[0]
                if isinstance(first, dict):
                    text = first.get("content") or first.get("text") or str(first)
                else:
                    text = str(first)
            else:
                text = str(out)
        else:
            text = str(out)
    parsed = robust_parse_json_like(text)
    if parsed.get("confidence") is None:
        parsed["confidence"] = 0.5
    return parsed

def file_id_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def read_edf_bytes(b: bytes):
    raw = mne.io.read_raw_edf(BytesIO(b), preload=True, verbose=False)
    return raw.get_data(), raw.info["sfreq"], raw.info["ch_names"]

def compute_psd(eeg: np.ndarray, sfreq: float, nperseg: int = 512):
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
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def char_accuracy(pred: str, true: str) -> Optional[float]:
    if not true: return None
    ed = levenshtein(pred, true)
    return max(0.0, 1.0 - ed / max(len(true), 1)) * 100.0

def word_accuracy(pred: str, true: str) -> Optional[float]:
    if not true: return None
    p = pred.split(); t = true.split()
    if len(t) == 0: return None
    matches = sum(1 for i in range(min(len(p), len(t))) if p[i] == t[i])
    return (matches / len(t)) * 100.0

# Explainability helpers
def example_model_score_from_probs(probs):
    arr = np.array(probs)
    return float(arr.max(axis=-1).mean())

def occlusion_importance(model_score_fn, base_input, step=4):
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

def permutation_importance(model_score_fn, base_input, n_iter=20):
    _, seq_len, channels = base_input.shape
    importance = np.zeros((channels, seq_len), dtype=float)
    baseline = model_score_fn(base_input)
    for _ in range(n_iter):
        perm = base_input.copy()
        for ch in range(channels):
            arr = perm[0, :, ch].copy()
            np.random.shuffle(arr)
            perm[0, :, ch] = arr
        score = model_score_fn(perm)
        importance += np.maximum(0.0, baseline - score)
    importance /= max(1, n_iter)
    total = importance.sum()
    if total > 0:
        importance /= total
    return importance

# ----------------- UI -----------------
st.title("EEG → Text & Speech (Explainable)")

# Provenance panel (mandatory visibility)
with st.expander("Provenance & Model Usage (required)", expanded=False):
    st.write("This application uses a third-party LLM (Gemini) to produce structured transcription estimates.")
    st.write("Gemini usage is cached per file to avoid repeated API calls. Provide your API key in Streamlit Secrets (GOOGLE_API_KEY) or as an environment variable.")
    st.write("If you need to remove external dependencies for deployment, consider converting/train your own ONNX models and remove the LLM usage.")

st.sidebar.markdown("System status")
st.sidebar.write({"gemini_client_installed": HAS_GENAI, "GOOGLE_API_KEY_present": bool(get_api_key())})

uploaded = st.file_uploader("Upload .edf file", type=["edf"], accept_multiple_files=False)
if not uploaded:
    st.info("Upload an EDF file to start.")
    st.stop()

file_bytes = uploaded.read()
fid = file_id_of_bytes(file_bytes)
st.sidebar.code(fid[:12])

# read EDF
try:
    eeg_data, sfreq, ch_names = read_edf_bytes(file_bytes)
except Exception as e:
    st.error(f"Failed to read EDF: {e}")
    st.stop()

st.success(f"Loaded EEG — channels: {len(ch_names)}, samples: {eeg_data.shape[1]}, sfreq: {sfreq:.2f} Hz")

# plots: raw & PSD
st.subheader("Raw EEG (first channels)")
nplot = min(6, eeg_data.shape[0])
fig_raw, axs = plt.subplots(nplot, 1, figsize=(10, 1.1 * nplot), sharex=True)
times = np.arange(eeg_data.shape[1]) / sfreq
for i in range(nplot):
    axs[i].plot(times, eeg_data[i], linewidth=0.6)
    axs[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
axs[-1].set_xlabel("Time (s)")
st.pyplot(fig_raw)

psd, freqs = compute_psd(eeg_data, sfreq)
st.subheader("Power Spectral Density (sample channels)")
fig_psd, ax = plt.subplots(figsize=(9, 3))
for i in range(min(4, psd.shape[0])):
    ax.semilogy(freqs, psd[i], label=ch_names[i])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend(fontsize="small")
st.pyplot(fig_psd)

# Prepare model input
psd_T = psd.T
psd_T = (psd_T - psd_T.mean()) / (psd_T.std() + 1e-9)
model_input = psd_T[None, ...].astype(np.float32)
st.write("Model input shape:", model_input.shape)

# Try to run optional ONNX models if present, else fallback
enc_sess = dec_sess = None
if os.path.exists("encoder.onnx"):
    try:
        enc_sess = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
        st.sidebar.success("encoder.onnx loaded")
    except Exception as e:
        st.sidebar.warning(f"encoder.onnx load failed: {e}")
if os.path.exists("decoder.onnx"):
    try:
        dec_sess = ort.InferenceSession("decoder.onnx", providers=["CPUExecutionProvider"])
        st.sidebar.success("decoder.onnx loaded")
    except Exception as e:
        st.sidebar.warning(f"decoder.onnx load failed: {e}")

# run encoder/decoder or deterministic fallback
encoded = None
if enc_sess:
    try:
        inp = enc_sess.get_inputs()[0].name
        encoded = enc_sess.run(None, {inp: model_input})[0]
        st.info("ONNX encoder ran.")
    except Exception as e:
        st.warning(f"ONNX encoder run failed: {e}")
if encoded is None:
    rng = np.random.RandomState(42)
    channels = model_input.shape[2]
    W = rng.randn(channels, 128).astype(np.float32)
    encoded = (model_input[0] @ W)[None, ...]
    st.info("Fallback encoder used.")

decoded_probs = None
if dec_sess:
    try:
        inp = dec_sess.get_inputs()[0].name
        decoded_probs = dec_sess.run(None, {inp: encoded})[0]
        st.info("ONNX decoder ran.")
    except Exception as e:
        st.warning(f"ONNX decoder run failed: {e}")
if decoded_probs is None:
    vocab_size = 128
    rng = np.random.RandomState(123)
    Wd = rng.randn(encoded.shape[2], vocab_size).astype(np.float32)
    logits = encoded @ Wd
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    decoded_probs = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
    st.info("Fallback decoder used.")

pred_tokens = np.argmax(decoded_probs, axis=-1)[0].tolist()
pred_text_model = tokens_to_text(pred_tokens)
model_confidence = float(decoded_probs.max(axis=-1).mean())

# Build PSD summary for prompts
def build_psd_summary(psd_arr, freqs_arr, ch_names_list, top_k=4):
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
    band_avgs = {}
    for b,(lo,hi) in bands.items():
        idx = np.where((freqs_arr>=lo)&(freqs_arr<=hi))[0]
        band_avgs[b] = psd_arr[:, idx].mean(axis=1).round(6).tolist() if idx.size else [0.0]*psd_arr.shape[0]
    avg = np.array([np.mean([band_avgs[b][i] for b in band_avgs]) for i in range(psd_arr.shape[0])])
    top_idx = np.argsort(avg)[-top_k:][::-1]
    top_ch = [{"name": ch_names_list[int(i)], "avg_power": float(avg[int(i)])} for i in top_idx]
    return {"bands": {k: band_avgs[k][:6] for k in band_avgs}, "top_channels": top_ch, "sfreq": float(sfreq)}

psd_summary = build_psd_summary(psd, freqs, ch_names)

# Persistent LLM cache per-file
cache_path = os.path.join(CACHE_DIR, f"{fid}.json")
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache_obj = json.load(f)
    gemini_gt = cache_obj.get("gemini_gt", {})
    gemini_pred = cache_obj.get("gemini_pred", {})
    st.info("Loaded cached LLM outputs for this file.")
else:
    if not get_api_key():
        st.error("GOOGLE_API_KEY missing (Streamlit Secrets or env). Configure to enable Gemini outputs.")
        st.stop()
    if not HAS_GENAI:
        st.error("google.generativeai client not installed in environment.")
        st.stop()
    try:
        prompt_gt = "Given this PSD summary, return a single JSON object only with keys transcription/confidence/rationale. Be concise.\n\n" + json.dumps(psd_summary)
        prompt_pred = "Given this PSD summary, return a single JSON object only with keys transcription/confidence/rationale for a predicted decoding. Be concise.\n\n" + json.dumps(psd_summary)
        gemini_gt = gemini_structured(prompt_gt)
        gemini_pred = gemini_structured(prompt_pred)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"gemini_gt": gemini_gt, "gemini_pred": gemini_pred, "psd_summary": psd_summary, "ts": time.time()}, f, indent=2)
        st.success("Gemini outputs generated & cached.")
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        st.stop()

# Show outputs (main UI concise)
st.subheader("Decoded outputs (concise)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Model (ONNX / fallback)**")
    st.write(pred_text_model)
    st.write(f"Model confidence: {model_confidence:.3f}")
with c2:
    st.markdown("**LLM (structured)**")
    st.write("GT estimate:")
    st.write(gemini_gt.get("transcription",""))
    st.write(f"confidence: {gemini_gt.get('confidence')}")
    st.write("Prediction:")
    st.write(gemini_pred.get("transcription",""))
    st.write(f"confidence: {gemini_pred.get('confidence')}")

# Manual GT and metrics
manual_gt = st.text_area("Optional: paste human ground-truth (overrides LLM GT for metrics)", height=120)
gt_used = manual_gt.strip() if manual_gt.strip() else gemini_gt.get("transcription","")
pred_used = gemini_pred.get("transcription","")
char_acc_val = char_accuracy(pred_used, gt_used) if gt_used else None
word_acc_val = word_accuracy(pred_used, gt_used) if gt_used else None

st.subheader("Accuracy")
if gt_used:
    st.metric("Character accuracy (%)", f"{char_acc_val:.2f}" if char_acc_val is not None else "N/A")
    st.metric("Word accuracy (%)", f"{word_acc_val:.2f}" if word_acc_val is not None else "N/A")
else:
    st.info("No ground-truth available — paste human GT to compute metrics.")

# Explainability
st.subheader("Explainability")
def model_score_fn(inp):
    try:
        if enc_sess is not None and dec_sess is not None:
            enc_in = enc_sess.get_inputs()[0].name
            enc_out = enc_sess.run(None, {enc_in: inp.astype(np.float32)})[0]
            dec_in = dec_sess.get_inputs()[0].name
            dec_out = dec_sess.run(None, {dec_in: enc_out.astype(np.float32)})[0]
            return example_model_score_from_probs(dec_out)
    except Exception:
        pass
    return example_model_score_from_probs(decoded_probs)

with st.spinner("Computing occlusion importance..."):
    occl_map = occlusion_importance(model_score_fn, model_input.copy(), step=max(1, model_input.shape[1]//24))
with st.spinner("Computing permutation importance..."):
    perm_map = permutation_importance(model_score_fn, model_input.copy(), n_iter=20)

fig1, ax1 = plt.subplots(figsize=(10,4))
im1 = ax1.imshow(occl_map, aspect='auto', origin='lower', cmap='magma')
ax1.set_yticks(np.arange(len(ch_names))); ax1.set_yticklabels(ch_names, fontsize=8)
ax1.set_xlabel("Frequency bin index"); ax1.set_title("Occlusion importance")
plt.colorbar(im1, ax=ax1, label="relative importance")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10,4))
im2 = ax2.imshow(perm_map, aspect='auto', origin='lower', cmap='viridis')
ax2.set_yticks(np.arange(len(ch_names))); ax2.set_yticklabels(ch_names, fontsize=8)
ax2.set_xlabel("Frequency bin index"); ax2.set_title("Permutation importance")
plt.colorbar(im2, ax=ax2, label="relative importance")
st.pyplot(fig2)

# Band aggregation summary
bands = {"delta": (0.5,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30), "gamma": (30,45)}
band_scores = {}
for b,(lo,hi) in bands.items():
    idx = np.where((freqs>=lo)&(freqs<=hi))[0]
    band_scores[b] = float(occl_map[:, idx].sum()) if idx.size else 0.0
figb, axb = plt.subplots(figsize=(6,3))
axb.bar(list(band_scores.keys()), list(band_scores.values()))
axb.set_ylabel("Aggregate occlusion importance")
st.pyplot(figb)

# Automated summary
top_idx = np.argsort(occl_map.mean(axis=1))[-3:][::-1]
top_channels = [ch_names[int(i)] for i in top_idx]
dominant_band = max(band_scores.items(), key=lambda x: x[1])[0] if band_scores else None
st.subheader("Automated explainability summary")
st.write(f"Top channels: {', '.join(top_channels)}. Dominant band: {dominant_band}.")

# TTS (gTTS)
tts_text = pred_used or pred_text_model
if tts_text:
    try:
        tts = gTTS(tts_text)
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmpf.name)
        st.audio(tmpf.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS failed: {e}")

# Download report
report = {
    "file_id": fid,
    "filename": getattr(uploaded, "name", ""),
    "gemini_gt": gemini_gt,
    "gemini_pred": gemini_pred,
    "model_pred": {"text": pred_text_model, "conf": model_confidence},
    "metrics": {"char_acc": char_acc_val, "word_acc": word_acc_val},
    "psd_summary": psd_summary,
    "explainability": {"band_scores": band_scores, "top_channels": top_channels},
    "timestamp": time.time()
}
st.download_button("Download report (JSON)", data=json.dumps(report, indent=2), file_name=f"report_{fid[:8]}.json", mime="application/json")
st.caption("Transparent use of third-party LLM is required; provenance available in the Provenance panel.")
