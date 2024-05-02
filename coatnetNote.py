from sys import argv
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pytorchCoatnet import coatnet_0
import tensorflow as tf
import torch

def load_training_data(training_data_directory):
    mel_spectrograms = []
    labels = []
    for note_folder in os.listdir(training_data_directory):
        if os.path.isdir(os.path.join(training_data_directory, note_folder)):
            note_path = os.path.join(training_data_directory, note_folder)
            for file_name in os.listdir(note_path):
                if file_name.endswith('.npy'):  # assuming mel spectrograms are in .npy format
                    file_path = os.path.join(note_path, file_name)
                    mel_spectrogram = np.load(file_path)  # load mel spectrogram using NumPy
                    # Reshape or preprocess mel spectrogram if needed
                    mel_spectrograms.append(mel_spectrogram)
                    label = str(note_folder)[0]
                    print("Loaded note ", label, " with size ", mel_spectrogram.shape)
                    labels.append(label)  # use the folder name as the label for the note
    labels = LabelEncoder().fit_transform(labels)  # convert labels to integers
    return np.array(mel_spectrograms), labels

# Load testing data
def load_testing_data(testing_file_path):
    # Load mel spectrograms and corresponding labels from the testing file
    testing_mel_spectrograms = []
    testing_labels = []

    # Iterate through each file in the testing file path
    for file_name in os.listdir(testing_file_path):
        if file_name.endswith('.npy'):  # Check if the file is a numpy file
            file_path = os.path.join(testing_file_path, file_name)
            mel_spectrogram = np.load(file_path)  # Load mel spectrogram using NumPy
            # Print shape for debugging
            print("Loaded testing mel spectrogram:", file_path)
            print("Loaded testing mel spectrogram shape:", mel_spectrogram.shape)
            # Reshape or preprocess mel spectrogram if needed
            testing_mel_spectrograms.append(mel_spectrogram)
            # Extract label from the file name (assuming the label is in the file name)
            label = file_name.split('_')[0]  # Extract the label from the file name
            print("Test Label: ", label)
            testing_labels.append(label)  # Append the label to the list of labels
    testing_labels = LabelEncoder().fit_transform(testing_labels)  # Convert labels to integers
    return np.array(testing_mel_spectrograms), testing_labels
    
def train_epoch(model, data, labels):
    model.train()
    for mel_spectrogram, label in zip(data, labels):
        label = torch.tensor(label)
        model(torch.from_numpy(mel_spectrogram))

def valid_epoch(model, data, labels):
    model.eval()
    correct = 0
    total = 0
    for mel_spectrogram, label in zip(data, labels):
        label = torch.tensor(label)
        output = model(torch.from_numpy(mel_spectrogram))
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return correct / total

def main():
    if (len(argv) > 1):
        training_data_directory = argv[1]
    else:
        training_data_directory = 'mel_spectrograms_(128x321)'

    training_mel_spectrograms, training_labels = load_training_data(training_data_directory)
    testing_mel_spectrograms, testing_labels = load_testing_data('testing_output')

    print(type(training_mel_spectrograms))

    net = coatnet_0()
    
    for epoch in range(10):
        print(f"-- training epoch {epoch} --")
        train_epoch(net, training_mel_spectrograms, training_labels)
        accuracy = valid_epoch(net, testing_mel_spectrograms, testing_labels)

        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()