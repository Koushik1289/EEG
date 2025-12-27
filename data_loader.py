import mne
from config import *
from utils import *

def load_eeg():
    ensure_dir(DATA_DIR)
    download_file(EEG_URL, EEG_FILE)

    raw = mne.io.read_raw_edf(EEG_FILE, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(LOW_FREQ, HIGH_FREQ)

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    return data, sfreq
