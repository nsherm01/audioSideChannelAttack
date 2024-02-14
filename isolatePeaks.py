import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def detect_peaks(audio_signal, sr, threshold=0.6):
    # Normalize the audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    # print_mel_spec(audio_signal)

    peaks = []
    for i in range(1, len(audio_signal) - 1):
        if audio_signal[i] > threshold and audio_signal[i] > audio_signal[i - 1] and audio_signal[i] > audio_signal[i + 1]:
            peaks.append(i)

    # Split the audio signal into chunks at the peak indices
    chunks = np.split(audio_signal, peaks)

    # Save each chunk as a new WAV file
    for i, chunk in enumerate(chunks):
        sf.write(f'peak_{i}.wav', chunk, sr)

    print("returning peaks: ", peaks)
    return peaks

def isolate_sounds(wav_file):
    # Load the audio file
    y, sr = librosa.load(wav_file)
    print("y: ", y)
    print("sr: ", sr)

    # Detect peaks in the audio signal
    peaks = detect_peaks(y, sr)

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

    print("returning sound_segments: ", sound_segments)
    return sound_segments

def create_mel_spectrogram(sound_segments, output_image):
    print("in create_mel_spectrogram")
    print("length of sound_segments: ", len(sound_segments))
    sr = 22050
    print("sr: ", sr)
    # Generate mel spectrogram for each sound segment
    for i, segment in enumerate(sound_segments):
        print("in for loop, i = ", i)
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)

        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)


        # Plot the mel spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.title(f'Mel Spectrogram - Sound Segment {i+1}')
        plt.colorbar(format='%+2.0f dB')

        print("got here")

        # Save the plot as a PNG image
        plt.savefig(output_image, bbox_inches='tight', pad_inches=0.0)

        plt.show()

# def print_mel_spec(sound_segment):
#     print("printing mel spec")
#     # Compute the mel spectrogram
#     mel_spectrogram = librosa.feature.melspectrogram(y=sound_segment, sr=sr)

#     # Convert to log scale (dB)
#     log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

#     # Plot the mel spectrogram
#     plt.figure(figsize=(10, 5))
#     librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
#     plt.title(f'Mel Spectrogram - Sound Segment')
#     plt.colorbar(format='%+2.0f dB')
#     plt.show()

        
# Example usage:
def main():
    flag = True
    input_wav_file = 'file_example_WAV_2MG.wav'
    output_image_file = 'mel_spec_out' + {i+1} + '.png'
    sound_segments = isolate_sounds(input_wav_file)
    create_mel_spectrogram(sound_segments, output_image_file)

if __name__ == "__main__":
    main()