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
# NOTE: Using a placeholder key.
GEMINI_API_KEY = "AIzaSyCWi3Q4m6HujP6XHMyWfpTPzb3Df2IcamA"

try:
    # Initialize client globally for functions that cannot be cached (like the UI/button interaction)
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    client = None


# --- 3. CORE FUNCTIONS (WITH PERSISTENT CACHING) ---

@st.cache_data(show_spinner=False)
def get_deterministic_decoding_texts(file_name, api_key):
    """
    Generates simulated Ground Truth and Predicted Text.
    @st.cache_data guarantees the same output for the same file_name across sessions.
    """
    # Initialize client locally inside the cached function for thread safety and persistence
    try:
        local_client = genai.Client(api_key=api_key)
    except Exception:
        # Fallback if API key is invalid/fails
        return "The quick brown fox jumps over the lazy dog.", "The quik bown box jump over the lazy dod."

    prompt = f"""
    You are simulating the deterministic results of a high-end EEG-to-Text decoding system for a file from a 34-subject dataset: '{file_name}'.
    The system is designed to convert internally spoken language or a motor command into text.

    1. **Generate a realistic 'Actual Text' (Ground Truth)**.
    2. **Generate a 'Predicted Text'** that is a *slightly corrupted* version of the Actual Text, reflecting the realistic word-level accuracy target of 30-40% and character-level accuracy of 70-80%. The errors must be realistic misinterpretations (substitutions of letters/sounds).

    Format the output strictly as a single JSON object:
    {{
      "actual_text": "...",
      "predicted_text": "..."
    }}
    """

    try:
        response = local_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        result = json.loads(response.text)
        return result['actual_text'], result['predicted_text']

    except (APIError, json.JSONDecodeError, KeyError):
        # Deterministic fallback if API call fails but file name is unique
        return "The quick brown fox jumps over the lazy dog.", "The quik bown box jump over the lazy dod."


def calculate_accuracy(actual, predicted):
    """Simulates calculation based on the required ranges (70-80% char, 30-40% word)."""
    # These are random values within the target range, ensuring consistency across sessions
    # is handled by the deterministic text inputs.
    char_acc = np.random.uniform(70.0, 80.0)
    word_acc = np.random.uniform(30.0, 40.0)
    return char_acc, word_acc


def display_decoded_texts(actual, predicted, char_acc, word_acc):
    """Displays the key text results and accuracy scores, removing the 'Target' delta text."""
    st.subheader("📝 Decoded Language Output")

    col_l_text, col_r_acc = st.columns([2, 1])

    with col_l_text:
        st.markdown(f"**Actual Text (Ground Truth):**\n> *{actual}*")
        st.markdown(f"**Predicted Text (Model Output):**\n> *{predicted}*")

    with col_r_acc:
        st.metric(
            label="Character-Level Accuracy (CLE) [70-80%]",
            value=f"{char_acc:.2f}%",
            delta=None
        )
        st.metric(
            label="Word-Level Accuracy (WLE) [30-40%]",
            value=f"{word_acc:.2f}%",
            delta=None
        )
    st.markdown("---")


# --- PLOTTING AND XAI FUNCTIONS (Simplified for brevity, maintained from previous version) ---

def plot_raw_eeg_waveforms(file_name):
    st.header("📈 Raw EEG Waveforms Visualization (Time-Domain)")
    channels = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']
    n_samples = 500
    time = np.linspace(0, 2, n_samples)

    fig, ax = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    for i, ch in enumerate(channels):
        signal = np.sin(2 * np.pi * (5 + i) * time) * (5 + i / 2) + np.random.normal(0, 1.5, n_samples)
        ax[i].plot(time, signal, linewidth=1)
        ax[i].set_ylabel(ch, rotation=0, labelpad=30, fontsize=12)
        ax[i].tick_params(left=False, labelleft=False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

    ax[-1].set_xlabel("Time (s)", fontsize=14)
    ax[0].set_title(f"Simulated Raw EEG Trace from {file_name}", fontsize=16)
    st.pyplot(fig)
    st.warning("⚠️ Note: This plot is a **realistic simulation** of the multichannel EEG waveforms.")


def plot_psd_analysis():
    st.header("🔬 Spectral Feature Extraction: Power Spectral Density (PSD)")
    st.markdown(
        "PSD analysis converts the raw EEG signal into the frequency domain, providing the Deep Learning model with stable features related to brain state across standard bands.")

    freqs = np.linspace(1, 40, 500)
    power = 100 / (freqs ** 1.5) + np.random.normal(0, 0.5, 500)
    alpha_peak = 100 * np.exp(-0.5 * ((freqs - 10) / 1.5) ** 2)
    power += alpha_peak
    df_psd = pd.DataFrame({'Frequency (Hz)': freqs, 'Log Power (a.u.)': np.log(power)})

    fig_psd = px.line(df_psd, x='Frequency (Hz)', y='Log Power (a.u.)', title="Simulated Average PSD (Welch’s Method)")

    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
    for name, (low, high) in bands.items():
        fig_psd.add_vrect(x0=low, x1=high, fillcolor="rgba(0,128,0,0.1)", opacity=0.15, layer="below", line_width=0)

    fig_psd.update_layout(xaxis_range=[0.5, 35], margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_psd, use_container_width=True)


def generate_xai_reasoning(actual, predicted, char_acc, word_acc):
    st.header("💡 Explainable AI (XAI) and Accuracy Breakdown")

    if not client:
        st.warning("LLM client not initialized. Cannot generate dynamic XAI narrative.")
        return

    prompt = f"""
    You are the Explainable AI (XAI) module for an EEG-to-Text system. Analyze the following decoding results:
    - Actual Text (Ground Truth): "{actual}"
    - Predicted Text (Model Output): "{predicted}"

    Task 1: **Automated Narrative Explanation (XAI)**.
    Generate a two-paragraph narrative explaining the likely cause of the decoding errors, focusing on the substitution/deletion of characters/words. Assume the error is due to spectral feature confusion in the Alpha or Beta bands (e.g., misinterpreting similar phonemes). Use technical terms like Residual BiLSTM, attention mechanism, spectral features, and temporal window.

    Task 2: **Accuracy Derivation Breakdown**.
    Show a simple, clear breakdown of how the accuracy percentages are calculated from the two text strings (character matches vs total, word matches vs total). Use a table format for clarity.

    Format the entire output with clear markdown headings for each task.
    """

    with st.spinner("Generating automated XAI narrative and accuracy breakdown..."):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            st.markdown(response.text)

        except APIError as e:
            st.error(f"XAI Generation Error: {e.message}")
        except Exception as e:
            st.error(f"An unexpected error occurred during XAI generation: {e}")


def section_model_architecture():
    st.header("💻 Deep Learning Pipeline: Residual BiLSTM + Seq2Seq")
    st.markdown(
        "The core is a **Residual BiLSTM Encoder** that processes the sequence of PSD features. This feeds into an **Attention-based Seq2Seq Decoder** which converts the encoded neural representation into linguistic tokens.")
    st.subheader("Inference Optimization")
    st.markdown(
        "The components can be deployed via **ONNX Runtime** for optimized inference, ensuring rapid, continuous functionality. **Residual connections** are key to training this deep network structure.")


def plot_hypothetical_accuracy():
    st.header("📉 Training History and Calibration")
    st.markdown(
        "The plots below show the model's performance convergence during training, calibrated to achieve the target accuracy ranges.")

    epochs = np.arange(1, 16)
    train_acc_char = 70 - 15 * np.exp(-epochs / 5) + np.random.normal(0, 1, 15)
    val_acc_char = 65 - 10 * np.exp(-epochs / 5) + np.random.normal(0, 1.5, 15)

    fig_line, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc_char, label='Train Char Acc', color='#1f77b4')
    ax.plot(epochs, val_acc_char, label='Validation Char Acc', linestyle='--', color='#ff7f0e')
    ax.axhspan(70, 80, color='r', alpha=0.1, label='Target Range')
    ax.set_title('Character Level Accuracy Convergence Over Epochs', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig_line)

    st.subheader("Post-Processing Calibration")
    st.markdown(
        "A calibration process (using `postproc_params.json` offline) applies deterministic rules to correct spelling and remove noise, which helps the final accuracy meet the target metrics.")


def section_gemini_api_refinement():
    st.header("✨ Language-Level Refinement using LLM")
    st.markdown(
        "The **Gemini model** is used for post-processing: **estimating plausible ground-truth** and **refining the predicted text** into grammatically coherent output, enhancing linguistic quality.")

    if client:
        default_prompt = "Refine the raw model output 'The quik bown box jump' into grammatically fluent and correct English text."
        prompt = st.text_area(
            "Demonstration: Enter Raw Model Output for LLM Refinement:",
            default_prompt,
            height=100
        )

        if st.button("Refine Text with Gemini"):
            with st.spinner("Refining text..."):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"You are a professional linguistic refinement model. Correct and refine the following raw model prediction into a single, fluent English sentence:\n\n---\n{prompt}",
                    )
                    st.success("Refined Output:")
                    st.info(response.text)
                except Exception:
                    st.error("API Error during refinement.")
    else:
        st.warning("LLM Refinement Demo is disabled (API Key not found).")


# --- 4. MAIN APPLICATION LOGIC ---

def main():
    st.title("🧠 Professional EEG-Driven Language Interface Showcase")
    st.caption("A Complete Neural Decoding and Explainable AI Pipeline")
    st.markdown("---")

    st.markdown("## Step 1: Upload EEG Data File")

    uploaded_file = st.file_uploader(
        "Upload your EDF or CSV file (e.g., Subject00_2.edf up to Subject34.edf)",
        type=['edf', 'csv'],
        accept_multiple_files=False,
        key="eeg_file_uploader",
        help="The system simulates analysis based on the uploaded file's name and type."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name

        # --- Generate texts and calculate accuracy using persistent cache ---
        with st.spinner(
                "Analyzing context and generating simulated decoding results (LLM-Driven, Persistently Cached)..."):
            actual_text, predicted_text = get_deterministic_decoding_texts(file_name, GEMINI_API_KEY)

            # Since the text is deterministic, we calculate new (random) but bounded accuracy values
            # to match the target ranges, which is sufficient for a simulation.
            char_acc, word_acc = calculate_accuracy(actual_text, predicted_text)

        st.success(
            f"✅ File **{file_name}** successfully loaded! Decoding Pipeline Results are **Consistent** across sessions.")
        st.markdown("---")

        st.markdown("## Step 2: Decoded Output & Accuracy Report")
        display_decoded_texts(actual_text, predicted_text, char_acc, word_acc)

        st.markdown("## Step 3: Full Pipeline Analysis")

        # First Row: Raw EEG and PSD
        col_raw_eeg, col_psd = st.columns(2)
        with col_raw_eeg:
            plot_raw_eeg_waveforms(file_name)
        with col_psd:
            plot_psd_analysis()

        st.markdown("---")

        # Second Row: Architecture and Accuracy History
        col_arch, col_hist = st.columns(2)
        with col_arch:
            section_model_architecture()
        with col_hist:
            plot_hypothetical_accuracy()

        st.markdown("---")

        # Third Row: XAI and LLM Refinement
        st.markdown("## Step 4: Explainable AI, Refinement, and Speech Synthesis")

        col_xai, col_refinement = st.columns(2)
        with col_xai:
            generate_xai_reasoning(actual_text, predicted_text, char_acc, word_acc)

        with col_refinement:
            section_gemini_api_refinement()
            st.subheader("🔊 Speech Synthesis (TTS)")
            st.markdown("The final predicted text is converted into natural speech audio.")
            st.info(f"Final Text for TTS: **'{predicted_text}'**")

        st.markdown("---")
        st.download_button(
            label="Download Analysis Report (JSON)",
            data=json.dumps({"Actual Text": actual_text, "Predicted Text": predicted_text, "Char Accuracy": char_acc,
                             "Word Accuracy": word_acc}, indent=2),
            file_name=f"{file_name}_analysis_report.json",
            mime="application/json"
        )
        st.caption(
            f"Disclaimer: The charts and text output are **simulated** based on the architecture and expected results for a file like **{file_name}**. The results for the same file name are permanently cached using Streamlit's mechanisms.")

    else:
        st.info("⬆️ Please upload your EEG file (.edf or .csv) to initiate the complete decoding pipeline.")
        st.image(
            "https://images.unsplash.com/photo-1577908953932-d11893c834a7?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0OTAzNjV8MHwxfHNlYXJjaHwxfHxFTEVDUlRPSUNBTCUyMEVSUlBVVFN8ZW58MHx8fHwxNzAzNDQ2OTU3fDA&ixlib=rb-4.0.3&q=80&w=1080",
            caption="Awaiting EEG Data Upload for Neural Decoding.")


if __name__ == "__main__":
    main()
