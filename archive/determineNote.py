import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, utils
from tensorflow.keras.layers import Input
import coatnet

# Define parameters
input_shape = (128, 321, 1)  # Define the input shape for the model (mel spectrogram shape
num_classes = 10  # Define the number of different notes
epochs = 10  # Define the number of epochs for training
batch_size = 1  # Define the batch size for training

# Load training data
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
                    # Print shape for debugging
                    print("Loaded mel spectrogram shape:", mel_spectrogram.shape)
                    # Reshape or preprocess mel spectrogram if needed
                    mel_spectrograms.append(mel_spectrogram)
                    label = str(note_folder)[0]
                    print("Training Label: ", label)
                    labels.append(label)  # use the folder name as the label for the note
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

    return np.array(testing_mel_spectrograms), testing_labels


# Load training and testing data
training_data_directory = 'mel_spectrograms_(128x321)'
mel_spectrograms, labels = load_training_data(training_data_directory)


# Convert labels to categorical
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = utils.to_categorical(labels_encoded, num_classes)

# Build the model
model = models.Sequential()
# Make sure the size doesnt get too small starting at 128x11
model.add(Input(shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(mel_spectrograms, labels_categorical, epochs=epochs, batch_size=batch_size)

# Load testing data
testing_file_path = 'testing_folders/testing_20_top_keys'
testing_mel_spectrograms, testing_labels = load_testing_data(testing_file_path)

# Convert labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
encoded_testing_labels = label_encoder.fit_transform(testing_labels)

# Convert the encoded labels to one-hot encoded format
encoded_testing_labels_categorical = utils.to_categorical(encoded_testing_labels, num_classes)

# Check the shape of your true labels
print('Shape of encoded_testing_labels:', encoded_testing_labels.shape)

# Convert the encoded testing labels to one-hot encoded format
encoded_testing_labels_categorical = utils.to_categorical(encoded_testing_labels, num_classes)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(testing_mel_spectrograms, encoded_testing_labels_categorical)
print('Test accuracy:', test_acc)

# Make predictions on the testing data
predictions = model.predict(testing_mel_spectrograms)

# Check the shape of your model's output during evaluation
print('Shape of model predictions:', predictions.shape)
