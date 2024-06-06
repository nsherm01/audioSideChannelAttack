import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import soundfile as sf
import sys
import os

'''
librosaPeaks.py

This script takes in a WAV file and isolates the peaks of the audio file using the librosa library.
This is useful for isolating the individual keystrokes of a keyboard in a recording of spaced keystokes.
It then creates a mel spectrogram for each note and saves it to a folder. The mel spectrograms are saved as numpy files in the output folder.
These mel spectrograms will be used to train a neural network to classify the notes in another script.

We used this python script to isolate notes for both testing and training. When you run the script with the -test flag, make sure to
handle the output folder correctly and handcode the labels in the create_mel_spectrogram function.

librosaPeaks.py only isolates the peaks for one given WAV file. To isolate peaks for multiple WAV files (i.e. your training data for each keystoke), 
you can use the script processWavs.py to run this file multiple times.

It can be run in two ways:
1. Run the script with the -test flag to generate the mel spectrograms for the 45 key test audio file.
2. Run the script with the key, audio file, and output folder as arguments to generate the mel spectrograms for a custom audio file.

Usage:
    python3 librosaPeaks.py -test
    python3 librosaPeaks.py <key> <audiofile> <output_folder>

Arguments:
    -test: Flag to generate mel spectrograms for the 45 key test audio file.
    key: The key of the audio file.
    audiofile: The path to the audio file.
    output_folder: The folder to save the mel spectrograms.

'''
hop_length = 128

def main():
    if (len(sys.argv) == 2):
        if (sys.argv[1] == "-test"):
            key = "#"
            audiofile = "audiofiles/45keytest.wav"
            output_folder = "45_testing_output"
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

'''
showStereoWaveform(y, sr)

This is a helper function that displays the stereo waveform of the audio file.
This is meant to be used for debugging purposes.

Parameters:
    y: The audio time series.
    sr: The sample rate of the audio.

Returns:
    None
'''
def showStereoWaveform(y, sr):
    plt.figure()
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr, color="blue")
    plt.title('Stereo')
    plt.show()



'''
showNoteDetection(onset_strength, notes_frames, T, t)

This is a helper function that displays the onset strength of the audio file.
This is meant to be used for debugging purposes.

Parameters:
    onset_strength: The onset strength of the audio.
    notes_frames: The frames of the notes.
    T: The total time of the audio.
    t: The time array.

Returns:
    None
'''
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




'''
showMelSpectrogram(mel_spectrogram, sr)

This is a helper function that displays the mel spectrogram of the audio file.
This is meant to be used for debugging purposes.

Parameters:
    mel_spectrogram: The mel spectrogram.
    sr: The sample rate of the audio.

Returns:
    None
'''
def showMelSpectrogram(mel_spectrogram, sr):
    times = librosa.frames_to_time(range(mel_spectrogram.shape[1]), sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_coords=times, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram - Sound Segment')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout() 
    plt.show()    





'''
isolatePeaks(file, key)

This function isolates the peaks of the audio file using the librosa library.
It removes onsets that are too close together or too close to the end of the audio.
It then creates the output notes and saves them to the notes folder.

Parameters:
    file: The path to the audio file.
    key: The key of the audio file.

Returns:
    sr: The sample rate of the audio.
    notes_samples: The isolated notes of the audio.
'''
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
            directory = os.path.join("training_notes", key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            sf.write(os.path.join(directory, key + '_note' + str(i) + '.wav'), note, sr)

    # showNoteDetection(onset_strength, notes_frames, T, t)
        
    return sr, notes_samples




'''
create_mel_spectrogram(segment, sr, i, key, output_folder)

This function creates a mel spectrogram for the given segment and saves it to the output folder.
Be careful with the key parameter. You must hardcode the labels of the audio here. 
The * is a false detected note and the generated mel spectrogram will be ignored/deleted.

Parameters:
    segment: The segment of the audio.
    sr: The sample rate of the audio.
    i: The index of the segment.
    key: The key of the audio.
    output_folder: The folder to save the mel spectrogram.

Returns:
    None
'''
def create_mel_spectrogram(segment, sr, i, key, output_folder):

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=16)
    print("Creating Mel Spectrogram Number: ", i, " with shape: ", mel_spectrogram.shape)

    # Change this string to match the correct label for each keystroke. The * is a false detected note and the generated mel spectrogram will be ignored/deleted.
    test_notes_list = list("*PLMNKOIHBUYGVCTFDXRESZAWQHDTPAQJUYRCMP*JAZPIRV")

    if (key == "#"):
        key = test_notes_list[i]
    np.save(os.path.join(output_folder, key + '_mel_spectrogram' + str(i)), mel_spectrogram)

    #showMelSpectrogram(mel_spectrogram, sr)    


if __name__ == "__main__":
    main()