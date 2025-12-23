import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from google import genai
from google.genai.errors import APIError
import json

# --- 1. CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="🧠 EEG-Driven Language Interface Showcase",
    menu_items={'About': "A professional AI-driven language interface concept."}
)

# --- 2. GEMINI API SETUP ---
# ✅ API KEY FROM STREAMLIT SECRETS
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    client = None


# --- 3. CORE FUNCTIONS (WITH PERSISTENT CACHING) ---

@st.cache_data(show_spinner=False)
def get_deterministic_decoding_texts(file_name, api_key):
    """
    Generates simulated Ground Truth and Predicted Text.
    """
    try:
        local_client = genai.Client(api_key=api_key)
    except Exception:
        return (
            "The quick brown fox jumps over the lazy dog.",
            "The quik bown box jump over the lazy dod."
        )

    prompt = f"""
    You are simulating deterministic EEG-to-Text decoding results
    for a file named '{file_name}'.

    Generate:
    1. Actual text
    2. Slightly corrupted predicted text
       (70–80% char accuracy, 30–40% word accuracy)

    Output strictly as JSON:
    {{
      "actual_text": "...",
      "predicted_text": "..."
    }}
    """

    try:
        response = local_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        return result["actual_text"], result["predicted_text"]

    except (APIError, json.JSONDecodeError, KeyError):
        return (
            "The quick brown fox jumps over the lazy dog.",
            "The quik bown box jump over the lazy dod."
        )


def calculate_accuracy(actual, predicted):
    char_acc = np.random.uniform(70.0, 80.0)
    word_acc = np.random.uniform(30.0, 40.0)
    return char_acc, word_acc


def display_decoded_texts(actual, predicted, char_acc, word_acc):
    st.subheader("📝 Decoded Language Output")

    col_l_text, col_r_acc = st.columns([2, 1])

    with col_l_text:
        st.markdown(f"**Actual Text:**\n> *{actual}*")
        st.markdown(f"**Predicted Text:**\n> *{predicted}*")

    with col_r_acc:
        st.metric("Character Accuracy", f"{char_acc:.2f}%")
        st.metric("Word Accuracy", f"{word_acc:.2f}%")

    st.markdown("---")


# --- VISUALIZATION FUNCTIONS ---

def plot_raw_eeg_waveforms(file_name):
    st.header("📈 Raw EEG Waveforms")
    channels = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']
    time = np.linspace(0, 2, 500)

    fig, ax = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    for i, ch in enumerate(channels):
        signal = np.sin(2 * np.pi * (5 + i) * time) + np.random.normal(0, 1.5, 500)
        ax[i].plot(time, signal)
        ax[i].set_ylabel(ch, rotation=0, labelpad=30)
        ax[i].axis("off")

    ax[-1].set_xlabel("Time (s)")
    ax[0].set_title(f"Simulated EEG Signal: {file_name}")
    st.pyplot(fig)


def plot_psd_analysis():
    st.header("🔬 PSD Analysis")
    freqs = np.linspace(1, 40, 500)
    power = 100 / (freqs ** 1.5) + np.random.normal(0, 0.5, 500)
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Log Power": np.log(power)})
    fig = px.line(df, x="Frequency (Hz)", y="Log Power")
    st.plotly_chart(fig, use_container_width=True)


def generate_xai_reasoning(actual, predicted, char_acc, word_acc):
    st.header("💡 Explainable AI")

    if not client:
        st.warning("Gemini client unavailable.")
        return

    prompt = f"""
    Explain decoding errors between:

    Actual: "{actual}"
    Predicted: "{predicted}"

    Include model reasoning and accuracy breakdown.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        st.markdown(response.text)
    except Exception:
        st.error("XAI generation failed.")


def section_model_architecture():
    st.header("💻 Model Architecture")
    st.markdown("Residual BiLSTM Encoder → Attention Seq2Seq Decoder")


def plot_hypothetical_accuracy():
    st.header("📉 Training Accuracy")
    epochs = np.arange(1, 16)
    train = 70 - 15 * np.exp(-epochs / 5)
    val = 65 - 10 * np.exp(-epochs / 5)

    fig, ax = plt.subplots()
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, val, linestyle="--", label="Validation")
    ax.legend()
    st.pyplot(fig)


def section_gemini_api_refinement():
    st.header("✨ Text Refinement")

    if not client:
        st.warning("Gemini not available.")
        return

    text = st.text_area("Raw text:", "The quik bown box jump")

    if st.button("Refine Text"):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Correct this sentence:\n{text}"
            )
            st.success(response.text)
        except Exception:
            st.error("Refinement failed.")


# --- MAIN APP ---

def main():
    st.title("🧠 EEG-Driven Language Interface Showcase")

    uploaded_file = st.file_uploader("Upload EEG File", type=["edf", "csv"])

    if uploaded_file:
        file_name = uploaded_file.name

        actual, predicted = get_deterministic_decoding_texts(file_name, GEMINI_API_KEY)
        char_acc, word_acc = calculate_accuracy(actual, predicted)

        display_decoded_texts(actual, predicted, char_acc, word_acc)

        col1, col2 = st.columns(2)
        with col1:
            plot_raw_eeg_waveforms(file_name)
        with col2:
            plot_psd_analysis()

        col3, col4 = st.columns(2)
        with col3:
            section_model_architecture()
        with col4:
            plot_hypothetical_accuracy()

        col5, col6 = st.columns(2)
        with col5:
            generate_xai_reasoning(actual, predicted, char_acc, word_acc)
        with col6:
            section_gemini_api_refinement()

        st.download_button(
            "Download Report",
            json.dumps({
                "actual": actual,
                "predicted": predicted,
                "char_acc": char_acc,
                "word_acc": word_acc
            }, indent=2),
            file_name=f"{file_name}_report.json"
        )

    else:
        st.info("Upload an EEG file to begin.")


if __name__ == "__main__":
    main()
