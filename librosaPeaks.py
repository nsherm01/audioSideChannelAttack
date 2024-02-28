import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import soundfile as sf

hop_length = 128

def main():
    audiofile = "audiofiles/D4x10.wav"
    sr, notes = isolatePeaks(audiofile)
    for i, segment in enumerate(notes):
        # Image file name
        output_image_file = 'mel_spec_out' + str(i+1) + '.png'
        create_mel_spectrogram(segment, output_image_file, sr, i)

def isolatePeaks(file):
    y, sr = librosa.load(file)
    '''plt.figure()
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Stereo')

    plt.show()'''

    onset_detect = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    print(onset_detect)

    
    N = len(y)
    T = N/float(sr)
    t = np.linspace(0, T, len(onset_strength))

    # plt.figure(figsize=(14, 5))
    # plt.plot(t, onset_strength)
    # plt.xlabel('Time (sec)')
    # plt.xlim(xmin=0)
    # plt.ylim(0)
    # plt.show()
    

    peaks = librosa.util.peak_pick(onset_strength, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=5)
    print(peaks)

    # plt.figure(figsize=(14, 5))
    # plt.plot(t, onset_strength, label='Onset strength')
    # plt.vlines(t[peaks], 0, onset_strength.max(), color='r', alpha=0.8, label='Selected peaks')
    # plt.xlabel('Time (sec)')
    # plt.legend()
    # plt.xlim(0, T)
    # plt.ylim(0)
    # plt.show()

    onset_detect_samples = librosa.frames_to_samples(onset_detect, hop_length=hop_length)
    peak_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)
    print(onset_detect_samples)
    print(peak_samples)


    # Add end sample index
    onset_samples = np.append(onset_detect_samples, len(y))

    notes = []
    # Save each note as a separate .wav file
    for i in range(len(onset_samples) - 1):
        start_sample = onset_samples[i]
        end_sample = onset_samples[i + 1] - hop_length
        note = y[start_sample:end_sample]
        sf.write(f'note_{i}.wav', note, sr)
        notes.append(note)
    return sr, notes



def create_mel_spectrogram(segment, output_image, sr, i):
    print("Creating Mel Spectrogram Number: ", i)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
    np.save('D4_mel_spectrogram.npy' + str(i), mel_spectrogram)

    # Convert to log scale (dB)
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Convert the frames to time (in seconds)
    '''
    times = librosa.frames_to_time(range(mel_spectrogram.shape[1]), sr=sr)


    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_coords=times, x_axis='time', y_axis='mel')
    plt.title(f'Mel Spectrogram - Sound Segment {i+1}')
    plt.colorbar(format='%+2.0f dB')
     # Set the x-axis limits
    plt.xlim(0, hop_length / sr)  # Change these values to your desired range
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0.0)

    '''

if __name__ == "__main__":
    main()
    