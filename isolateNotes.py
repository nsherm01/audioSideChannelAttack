import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import find_peaks

def isolate_notes(input_file, output_prefix, window_size, threshold):
    # Load the WAV file
    sample_rate, data = wav.read(input_file)
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = data.sum(axis=1) / 2

    # Apply FFT to the audio data
    fft_data = np.abs(np.fft.fft(data))

    # Find peaks in the frequency spectrum
    peaks, _ = find_peaks(fft_data, height=threshold*np.max(fft_data), distance=window_size//2)

    # Create WAV files for each note
    for i, peak in enumerate(peaks):
        start = max(0, peak - window_size//2)
        end = min(len(data), peak + window_size//2)
        note_data = data[start:end]

        # Save the note as a WAV file
        output_file = f"{output_prefix}_note_{i}.wav"
        wav.write(output_file, sample_rate, note_data.astype(np.int16))
        print(f"Note {i+1} saved as {output_file}")

if __name__ == "__main__":
    input_file = 'audiofiles/scale.wav'  # Specify the input WAV file
    output_prefix = "output_note"  # Prefix for output WAV files
    window_size = 5000  # Window size for isolating notes
    threshold = 0.3  # Peak detection threshold

    isolate_notes(input_file, output_prefix, window_size, threshold)