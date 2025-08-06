import numpy as np
from scipy.signal import stft
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.signal import welch

# Generate dummy signal
fs = 48000
duration = 2.0
t_ax = np.linspace(0, duration, int(fs*duration), endpoint=False)
x = 0.1 * np.random.randn(len(t_ax))
mask = (t_ax >= 0.5) & (t_ax < 0.7)
x[mask] += 1.0 * np.sin(2 * np.pi * 1000 * t_ax[mask])

# Compute STFT
nperseg = 1024
noverlap = nperseg // 2
f, t_frames, Z = stft(x, fs, nperseg=nperseg, noverlap=noverlap)
S = np.abs(Z)

# Noise floor and activity index
noise_floor = np.median(S, axis=1, keepdims=True)
excess = np.maximum(0, S - noise_floor)
D = np.sum(excess, axis=0)
D_smooth = uniform_filter1d(D, size=3)

# Plot spectrogram
plt.figure()
plt.pcolormesh(t_frames, f, 20 * np.log10(S), shading='gouraud')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (dB)')
plt.colorbar(label='Magnitude (dB)')
plt.tight_layout()
plt.show()

# Plot activity index
plt.figure()
plt.plot(t_frames, D_smooth)
plt.xlabel('Time (s)')
plt.ylabel('Activity index')
plt.title('Narrow-band Activity Index')
plt.tight_layout()
plt.show()


def rms_noise_welch(x, fs, fmin=1e3, fmax=50e3, nperseg=1024, noverlap=None):
    """
    Estimate broadband noise RMS over a specified frequency band using Welch's method.
    """
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    mask = (f >= fmin) & (f <= fmax)
    P_noise = np.trapz(Pxx[mask], f[mask])
    return np.sqrt(P_noise)

def rms_over_time(x, fs, window_sec=1.0, step_sec=0.5, **welch_kwargs):
    """
    Compute RMS noise in sliding windows over time.
    
    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz).
    window_sec : float
        Window length in seconds.
    step_sec : float
        Step size between windows in seconds.
    welch_kwargs : dict
        Additional keyword args passed to rms_noise_welch.
        
    Returns
    -------
    times : ndarray
        Time stamps at the center of each window (s).
    sigmas : ndarray
        RMS noise values for each window.
    """
    window_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    sigmas = []
    times = []
    for start in range(0, len(x) - window_len + 1, step_len):
        segment = x[start:start + window_len]
        sigma = rms_noise_welch(segment, fs, **welch_kwargs)
        center_time = (start + window_len / 2) / fs
        sigmas.append(sigma)
        times.append(center_time)
    return np.array(times), np.array(sigmas)

# Example usage:
times, sigmas = rms_over_time(x, fs=fs, window_sec=0.05, step_sec=0.1, nperseg=512)
plt.figure()
plt.plot(times, sigmas)
plt.xlabel("Time (s)")
plt.ylabel("RMS Noise (V)")
plt.title("RMS Broadband Noise vs Time")
plt.show()