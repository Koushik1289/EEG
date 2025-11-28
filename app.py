import streamlit as st
import numpy as np
import mne
from scipy.signal import welch
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from einops import rearrange

# -------------------------------------------------------
# 1) STREAMLIT UI
# -------------------------------------------------------
st.title("EEG → PSD → Residual BiLSTM → Seq2Seq → Agentic AI")
st.write("Upload EEG (.edf) file and run complete pipeline")

uploaded = st.file_uploader("Upload .edf EEG file", type=["edf"])

# -------------------------------------------------------
# 2) Function: extract EEG & compute PSD
# -------------------------------------------------------
def compute_psd(eeg, sfreq):
    psd_list = []
    for ch in eeg:
        f, pxx = welch(ch, sfreq, nperseg=512)
        psd_list.append(pxx)
    return np.array(psd_list), f

# -------------------------------------------------------
# 3) Residual BiLSTM model (example architecture)
# -------------------------------------------------------
def build_residual_bilstm(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inp)
    res = x
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
    return tf.keras.Model(inp, x)

# -------------------------------------------------------
# 4) Simple Seq2Seq Decoder (example)
# -------------------------------------------------------
def build_seq2seq_decoder():
    decoder_in = tf.keras.Input(shape=(None, 64))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(decoder_in)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(30, activation='softmax'))(x)
    return tf.keras.Model(decoder_in, out)

# -------------------------------------------------------
# 5) XAI (SHAP)
# -------------------------------------------------------
def explain_model(model, sample):
    explainer = shap.DeepExplainer(model, np.expand_dims(sample, 0))
    shap_values = explainer.shap_values(np.expand_dims(sample, 0))
    return shap_values

# -------------------------------------------------------
# 6) Agentic AI (LLM reasoning layer)
# -------------------------------------------------------
def agentic_interpretation(text):
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyAtX1QJdv-y5xasT3elZ-fqQiPZUT8kwpY")

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"""You are an agentic AI. 
        Given this EEG decoded output: {text}
        - Interpret the meaning
        - Detect anomalies
        - Suggest improvements
        - Self-correct if inconsistencies exist"""
    )
    return response.text

# -------------------------------------------------------
# 7) PIPELINE EXECUTION
# -------------------------------------------------------
if uploaded:
    st.subheader("📥 Reading EEG File...")
    raw = mne.io.read_raw_edf(BytesIO(uploaded.read()), preload=True)
    eeg_data = raw.get_data()
    sfreq = raw.info['sfreq']
    st.success("EEG Loaded Successfully")

    st.subheader("📊 Computing Power Spectral Density (PSD)...")
    psd, freqs = compute_psd(eeg_data, sfreq)
    st.line_chart(psd[0][:200])

    # reshape for model
    psd_input = rearrange(psd, "c f -> f c")[None, :, :]

    st.subheader("🧠 Running Residual BiLSTM Encoder...")
    encoder = build_residual_bilstm((psd_input.shape[1], psd_input.shape[2]))
    encoded_features = encoder.predict(psd_input)
    st.success("Residual BiLSTM encoding completed")

    st.subheader("🔁 Running Seq2Seq Decoder...")
    decoder = build_seq2seq_decoder()
    decoded_output = decoder.predict(encoded_features)
    predicted_tokens = np.argmax(decoded_output[0], axis=-1)
    st.write("Decoded sequence:", predicted_tokens)

    st.subheader("🪄 Explainability (XAI)")
    if st.button("Generate SHAP Explanation"):
        shap_values = explain_model(encoder, psd_input[0])
        st.write("SHAP Explanation Generated")

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[0], psd_input[0], show=False)
        st.pyplot(fig)

    st.subheader("🤖 Agentic AI Interpretation")
    if st.button("Interpret with Agentic AI"):
        explanation = agentic_interpretation(str(predicted_tokens))
        st.write(explanation)

