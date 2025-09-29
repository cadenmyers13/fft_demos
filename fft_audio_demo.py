import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Parameters
samplerate = 44100
duration = 15

# Record audio
print("Recording...")
voice = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float64')
sd.wait()
print("Done recording!")
voice = voice.flatten()

# Time axis
t = np.linspace(0, duration, len(voice))

# FFT
N = len(voice)
freqs = np.fft.rfftfreq(N, 1/samplerate)
fft_vals = np.fft.rfft(voice)
magnitude = np.abs(fft_vals) / N

# --- Plot time and frequency plots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Time-domain waveform
ax1.plot(t, voice)
ax1.set_title("Voice Signal (Time Domain)")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")

# Frequency spectrum
line, = ax2.plot(freqs, magnitude)
ax2.set_xlim(0, 2000)  # focus on human voice
ax2.set_title("Frequency Spectrum (Click to play frequency)")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Magnitude")

# Click event handler
def onclick(event):
    if event.inaxes != ax2:
        return
    freq_clicked = event.xdata
    if freq_clicked is None:
        return
    
    print(f"Playing frequency: {freq_clicked:.1f} Hz")
    # Generate sine wave for 1 second
    t_tone = np.linspace(0, 1, samplerate, endpoint=False)
    tone = np.sin(2 * np.pi * freq_clicked * t_tone)
    tone /= np.max(np.abs(tone))  # normalize
    sd.play(tone, samplerate=samplerate)
    sd.wait()

# Connect click event to spectrum plot
cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.tight_layout()
plt.show()
