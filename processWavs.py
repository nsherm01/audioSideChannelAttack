import os
import subprocess
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import soundfile as sf
from skimage.transform import resize

# Define paths
input_folder = 'top_keyboard_notes'
output_folder = 'mel_spectrograms_NOTsized'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of WAV files
wav_files = [file for file in os.listdir(input_folder) if file.endswith('.wav')]

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
