import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import soundfile as sf
import sys
import os

hop_length = 128

def main():
    print(len(sys.argv))
    if (len(sys.argv) == 2):
        if (sys.argv[1] == "-test"):
            key = "#"
            audiofile = "audiofiles/Top_Row_Test_Data.wav"
            output_folder = "testing_output"
    elif (len(sys.argv) == 4):
        key = sys.argv[1]
        audiofile = sys.argv[2]
        output_folder = sys.argv[3]
    else:
        print("Invalid number of arguments")
        return
        
    
    # Returns an array of tuples: (note_begin, note_end)
    sr, notes = isolatePeaks(audiofile, key)

    # Loop over each note and create a mel spectrogram
    for i, segment in enumerate(notes):
        create_mel_spectrogram(segment, sr, i, key, output_folder)

def showStereoWaveform(y, sr):
    plt.figure()
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr, color="blue")
    plt.title('Stereo')
    plt.show()

def showNoteDetection(onset_strength, notes_frames, T, t):

    note_begins = [note[0] for note in notes_frames]
    note_ends = [note[1] for note in notes_frames]
    
    peaks = librosa.util.peak_pick(onset_strength, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=1.5, wait=40)

    plt.figure(figsize=(14, 5))
    plt.plot(t, onset_strength, label='Onset strength')
    plt.vlines(t[peaks], 0, onset_strength.max(), color='r', alpha=0.8, label='Selected peaks')
    plt.vlines(t[note_ends], 0, onset_strength.max(), color='g', alpha=0.8, label='Note Ends')
    plt.vlines(t[note_begins], 0, onset_strength.max(), color='y', alpha=0.8, label='Note Begins')
    plt.xlabel('Time (sec)')
    plt.legend()
    plt.xlim(0, T) 
    plt.ylim(0)
    plt.show()

def showMelSpectrogram(mel_spectrogram, sr):
    times = librosa.frames_to_time(range(mel_spectrogram.shape[1]), sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_coords=times, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram - Sound Segment ' + str(i))
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout() 
    plt.show()    


def isolatePeaks(file, key):
    y, sr = librosa.load(file)

    # showStereoWaveform(y, sr)

    onset_detect = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    N = len(y)
    T = N/float(sr)
    t = np.linspace(0, T, len(onset_strength))

    # Remove onsets that are too close together or too close to the end of the audio
    # Create each note to be 40 frames long
    notes_frames = [(onset_detect[0], onset_detect[0] + 40)]
    for i in range(1, len(onset_detect)):
        if onset_detect[i] - onset_detect[i-1] > 50 and onset_detect[i] + 40 < len(t):
            notes_frames.append((onset_detect[i], onset_detect[i] + 40))

    # Convert the note segments into samples
    note_begins = librosa.frames_to_samples([note[0] for note in notes_frames], hop_length=hop_length)
    note_ends = librosa.frames_to_samples([note[1] for note in notes_frames], hop_length=hop_length)

    # Create the output notes
    notes_samples = [y[sample_begin:sample_end]
                    for sample_begin, sample_end
                    in zip(note_begins, note_ends)]
        
    # If doing training, save the notes to the notes folder
    if (key != "#"):
        for i, note in enumerate(notes_samples):
            sf.write(os.path.join("training_notes", key + '_note' + str(i) + '.wav'), note, sr)

    showNoteDetection(onset_strength, notes_frames, T, t)
        
    return sr, notes_samples


def create_mel_spectrogram(segment, sr, i, key, output_folder):

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=16)
    print("Creating Mel Spectrogram Number: ", i, " with shape: ", mel_spectrogram.shape)

    test_notes_list = ['DELETE','Y','E','O','U','P','I','U','U','T','Y','E','Q','U','E','W','T','Q','I','Y','P']

    if (key == "#"):
        key = test_notes_list[i]
    np.save(os.path.join(output_folder, key + '_mel_spectrogram' + str(i)), mel_spectrogram)

    # showMelSpectrogram(mel_spectrogram, sr)    


if __name__ == "__main__":
    main()