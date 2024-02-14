import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav_to_mel_spectrogram(wav_file, output_image):
    # Load the audio file
    y, sr = librosa.load(wav_file)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the mel spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# Example usage:
input_wav_file = 'file_example_WAV_2MG.wav'
output_image_file = 'mel_spectrogram.png'
wav_to_mel_spectrogram(input_wav_file, output_image_file)
