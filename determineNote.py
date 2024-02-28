import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, utils

# Define parameters
input_shape = (128, 12, 1)  # Define your mel spectrogram shape
num_classes = 8  # Define the number of different notes
epochs = 10  # Define the number of epochs for training
batch_size = 32  # Define the batch size for training

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
                    labels.append(note_folder)  # use the folder name as the label for the note
    return np.array(mel_spectrograms), labels

# Load testing data
def load_testing_data(testing_file_path):
    # Load mel spectrograms and corresponding labels from the testing file
    testing_mel_spectrograms = []
    testing_labels = []
    # Implement your code to load the testing data here
    return testing_mel_spectrograms, testing_labels

# Load training and testing data
training_data_directory = 'training_data(npy_mel_spec)'
mel_spectrograms, labels = load_training_data(training_data_directory)

# Convert labels to categorical
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = utils.to_categorical(labels_encoded, num_classes)

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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
testing_file_path = 'path/to/your/testing_file'
testing_mel_spectrograms, testing_labels = load_testing_data(testing_file_path)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(testing_mel_spectrograms, testing_labels)
print('Test accuracy:', test_acc)

# Make predictions on the testing data
predictions = model.predict(testing_mel_spectrograms)
