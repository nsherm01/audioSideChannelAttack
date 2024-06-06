import os
import subprocess
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import soundfile as sf
from skimage.transform import resize

'''
processWavs.py

This script processes multiple WAV files using librosaPeaks.py. It takes in a folder of WAV files and processes each one using librosaPeaks.py.
This script is useful for processing multiple WAV files at once. It is used to generate the mel spectrograms for the training data. 
Without this script, you would have to run librosaPeaks.py for each WAV file individually.

Before use, ensure you have a folder of WAV files to process. The WAV files should be named with the note they represent (e.g. A.wav, B.wav, etc.).
Each WAV file should contain only the single note it represents (i.e. 25 keystokes of the 'A' key for A.wav).

Usage:
    python3 processWavs.py

Arguments:
    None

'''
# Define paths
input_folder = 'keyboard_notes'
output_folder = 'mel_spectrograms_(128x321)'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of WAV files
wav_files = [file for file in os.listdir(input_folder) if file.endswith('.wav')]

print(wav_files)

# Loop through each WAV file
for wav_file in wav_files:
    note = wav_file[0]
    input_path = os.path.join(input_folder, wav_file)
    print("Processing Input Path: ", input_path)
    output_subfolder = os.path.join(output_folder, os.path.splitext(wav_file)[0])

    # Create subfolder for output if it doesn't exist
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Run LibrosaPeaks.py using subprocess
    subprocess.run(['python3', 'LibrosaPeaks.py', note, input_path, output_subfolder])
