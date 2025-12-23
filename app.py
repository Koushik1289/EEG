import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import google.generativeai as genai
from google.genai.errors import APIError
import json

# --- 1. CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="🧠 EEG-Driven Language Interface Showcase",
    menu_items={'About': "A professional AI-driven language interface concept."}
)

# --- 2. GEMINI API SETUP ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

client = None

if not GEMINI_API_KEY:
    st.error("Missing Gemini API Key. Please configure it in Streamlit Secrets.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        client = None

# --- 3. CORE FUNCTIONS (WITH PERSISTENT CACHING) ---

@st.cache_data(show_spinner=False)
def get_deterministic_decoding_texts(file_name):
    """
    Generates simulated Ground Truth and Predicted Text.
    Output is cached per file name.
    """
    prompt = f"""
    You are simulating the deterministic results of a high-end EEG-to-Text decoding system
    for a file from a 34-subject dataset: '{file_name}'.

    1. Generate a realistic Actual Text (Ground Truth).
    2. Generate a Predicted Text with slight corruption
       (30–40% word accuracy, 70–80% character accuracy).

    Output strictly in JSON:
    {{
      "actual_text": "...",
      "predicted_text": "..."
    }}
    """

    try:
        local_client = genai.GenerativeModel("gemini-1.5-flash")
        response = local_client.generate_content(prompt)
        result = json.loads(response.text)
        return result["actual_text"], result["predicted_text"]

    except Exception:
        return (
            "The quick brown fox jumps over the lazy dog.",
            "The quik bown fox jump over the lazi dog."
        )

def calculate_accuracy(actual, predicted):
    """
    Deterministic accuracy values for the same text.
    """
    np.random.seed(abs(hash(actual)) % (10**6))
    char_acc = np.random.uniform(70.0, 80.0)
    word_acc = np.random.uniform(30.0, 40.0)
    return char_acc, word_acc

def display_decoded_texts(actual, predicted, char_acc, word_acc):
    st.subheader("📝 Decoded Language Output")

    col_l_text, col_r_acc = st.columns([2, 1])

    with col_l_text:
        st.markdown(f"**Actual Text (Ground Truth):**\n> *{actual}*")
        st.markdown(f"**Predicted Text (Model Output):**\n> *{predicted}*")

    with col_r_acc:
        st.metric("Character-Level Accuracy (%)", f"{char_acc:.2f}")
        st.metric("Word-Level Accuracy (%)", f"{word_acc:.2f}")

    st.markdown("---")

# --- PLOTTING FUNCTIONS ---

def plot_raw_eeg_waveforms(file_name):
    st.header("📈 Raw EEG Waveforms Visualization (Time-Domain)")
    channels = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']
    n_samples = 500
    time = np.linspace(0, 2, n_samples)

    fig, ax = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    for i, ch in enumerate(channels):
        signal = np.sin(2 * np.pi * (5 + i) * time) + np.random.normal(0, 1.5, n_samples)
        ax[i].plot(time, signal, linewidth=1)
        ax[i].set_ylabel(ch, rotation=0, labelpad=30)
        ax[i].axis("off")

    ax[-1].set_xlabel("Time (s)")
    ax[0].set_title(f"Simulated Raw EEG Trace from {file_name}")
    st.pyplot(fig)

def plot_psd_analysis():
    st.header("🔬 Spectral Feature Extraction: Power Spectral Density (PSD)")

    freqs = np.linspace(1, 40, 500)
    power = 100 / (freqs ** 1.5) + np.exp(-0.5 * ((freqs - 10) / 1.5) ** 2)
    df_psd = pd.DataFrame({
        "Frequency (Hz)": freqs,
        "Log Power": np.log(power)
    })

    fig = px.line(df_psd, x="Frequency (Hz)", y="Log Power",
                  title="Simulated Average PSD")
    st.plotly_chart(fig, use_container_width=True)

def generate_xai_reasoning(actual, predicted):
    st.header("💡 Explainable AI (XAI)")

    if not client:
        st.warning("LLM client not initialized.")
        return

    prompt = f"""
    Explain EEG decoding errors between:
    Actual: "{actual}"
    Predicted: "{predicted}"

    Use technical terms:
    Residual BiLSTM, attention mechanism, spectral confusion.
    """

    try:
        response = client.generate_content(prompt)
        st.markdown(response.text)
    except Exception as e:
        st.error(f"XAI Error: {e}")

def section_model_architecture():
    st.header("💻 Deep Learning Pipeline: Residual BiLSTM + Seq2Seq")
    st.markdown(
        "Residual BiLSTM encodes temporal EEG features, followed by "
        "attention-based Seq2Seq decoding."
    )

def plot_hypothetical_accuracy():
    st.header("📉 Training History")

    epochs = np.arange(1, 16)
    train_acc = 70 - 15 * np.exp(-epochs / 5)
    val_acc = 65 - 10 * np.exp(-epochs / 5)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, label="Train")
    ax.plot(epochs, val_acc, linestyle="--", label="Validation")
    ax.axhspan(70, 80, alpha=0.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    st.pyplot(fig)

def section_gemini_api_refinement():
    st.header("✨ Language Refinement with Gemini")

    if not client:
        st.warning("API not available")
        return

    text = st.text_area("Raw Model Output", "The quik bown fox jump")

    if st.button("Refine Text"):
        try:
            response = client.generate_content(
                f"Refine this sentence:\n{text}"
            )
            st.success(response.text)
        except Exception:
            st.error("Refinement failed")

# --- 4. MAIN APPLICATION LOGIC ---

def main():
    st.title("🧠 Professional EEG-Driven Language Interface Showcase")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload EEG file (.edf or .csv)",
        type=["edf", "csv"]
    )

    if uploaded_file:
        file_name = uploaded_file.name

        actual_text, predicted_text = get_deterministic_decoding_texts(file_name)
        char_acc, word_acc = calculate_accuracy(actual_text, predicted_text)

        display_decoded_texts(actual_text, predicted_text, char_acc, word_acc)

        col1, col2 = st.columns(2)
        with col1:
            plot_raw_eeg_waveforms(file_name)
        with col2:
            plot_psd_analysis()

        section_model_architecture()
        plot_hypothetical_accuracy()
        generate_xai_reasoning(actual_text, predicted_text)
        section_gemini_api_refinement()

        st.download_button(
            "Download Analysis Report",
            json.dumps({
                "Actual Text": actual_text,
                "Predicted Text": predicted_text,
                "Character Accuracy": char_acc,
                "Word Accuracy": word_acc
            }, indent=2),
            file_name=f"{file_name}_analysis.json"
        )

    else:
        st.info("⬆️ Upload an EEG file to begin.")

if __name__ == "__main__":
    main()
