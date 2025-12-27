import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.title("training loss")
    plt.show()

def plot_eeg(data):
    for i in range(min(6, data.shape[0])):
        plt.plot(data[i][:2000] + i*50)
    plt.title("raw eeg")
    plt.show()

def plot_psd(data, sfreq):
    for i in range(3):
        freqs, psd = welch(data[i], fs=sfreq)
        plt.plot(freqs, psd)
    plt.title("psd")
    plt.show()

def plot_importance(scores):
    plt.plot(scores)
    plt.title("feature importance")
    plt.show()
