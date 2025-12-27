DATA_DIR = "openneuro_data"
EEG_FILE = f"{DATA_DIR}/sub-01_eeg.edf"
EEG_URL = "https://s3.amazonaws.com/openneuro.org/ds002778/sub-01/eeg/sub-01_task-rest_eeg.edf"

LOW_FREQ = 0.5
HIGH_FREQ = 40.0

WINDOW_SEC = 1.0
OVERLAP = 0.75

SEQ_LEN = 12
EPOCHS = 250
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

MODEL_NAME = "eeg_model.h5"
