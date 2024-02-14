import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def detect_peaks(audio_signal, threshold=0.5):
    peaks = []
    for i in range(1, len(audio_signal) - 1):
        if audio_signal[i] > threshold and audio_signal[i] > audio_signal[i - 1] and audio_signal[i] > audio_signal[i + 1]:
            peaks.append(i)
    return peaks

def isolate_sounds(wav_file):
    # Load the audio file
    y, sr = librosa.load(wav_file)

    # Detect peaks in the audio signal
    peaks = detect_peaks(y)

    # Segment the audio around each peak
    sound_segments = []
    for peak_index in peaks:
        # Define segment boundaries around the peak
        segment_start = max(0, peak_index - 22050)  # 1 second before the peak (assuming 22050 samples/second)
        segment_end = min(len(y), peak_index + 22050)  # 1 second after the peak

        # Extract the segment
        segment = y[segment_start:segment_end]

        # Store the segment
        sound_segments.append(segment)

    return sound_segments

def create_mel_spectrogram(sound_segments):
    # Generate mel spectrogram for each sound segment
    for i, segment in enumerate(sound_segments):
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)

        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Plot the mel spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.title(f'Mel Spectrogram - Sound Segment {i+1}')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

# Example usage:
input_wav_file = 'file_example_WAV_2MG.wav'
sound_segments = isolate_sounds(input_wav_file)
create_mel_spectrogram(sound_segments)
