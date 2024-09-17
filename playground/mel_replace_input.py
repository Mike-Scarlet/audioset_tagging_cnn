
import librosa
from matplotlib import pyplot as plt
import numpy as np

aud_path = "resources/R9_ZSCveAHg_7s.wav"
sample_rate = 32000

window_size = 1024
n_fft = window_size
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
power = 2.0

waveform, _ = librosa.core.load(aud_path, sr=sample_rate, mono=True)

""" check stft same """
stft_result = librosa.stft(
  waveform, n_fft=n_fft, hop_length=hop_size, 
  win_length=window_size, window="hann", center=True, pad_mode="reflect")

S = (
      np.abs(stft_result)
      ** power
)
pass


mel_spec = librosa.feature.melspectrogram(
  y=waveform, sr=sample_rate, n_fft=n_fft, win_length=window_size, hop_length=hop_size,
  window="hann", center=True, power=power,
  n_mels=mel_bins, fmin=fmin, fmax=fmax, pad_mode="reflect")

mel_spec_power = librosa.power_to_db(mel_spec)
mel_spec_power[mel_spec_power > 10] = 10
mel_spec_power[mel_spec_power < -60] = -60

# (64, 701)

plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(mel_spec,ref=np.max),y_axis='mel', fmax=fmax, x_axis='time')
librosa.display.specshow(mel_spec_power,y_axis='mel', fmax=fmax, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

if __name__ == "__main__":
  pass