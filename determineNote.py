import numpy as np
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Define parameters
input_shape = (time_steps, freq_bins, 1)  # Define your mel spectrogram shape
num_classes = number_of_notes  # Define the number of different notes

# Load training data
def load_training_data(training_data_directory):
    mel_spectrograms = []
    labels = []
    for note_folder in os.listdir(training_data_directory):
        if os.path.isdir(os.path.join(training_data_directory, note_folder)):
            note_path = os.path.join(training_data_directory, note_folder)
            for file_name in os.listdir(note_path):
                if file_name.endswith('.png'):  # assuming mel spectrograms are in PNG format
                    file_path = os.path.join(note_path, file_name)
                    mel_spectrogram = load_mel_spectrogram(file_path)  # load mel spectrogram using your method
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
training_data_directory = 'training_data'
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
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test, y_test), callbacks=callbacks_list)

# Load testing data
testing_file_path = 'path/to/your/testing_file'
testing_mel_spectrograms, testing_labels = load_testing_data(testing_file_path)

# Evaluate the model on the testing data
saved_model = load_model('best_model.h5')
test_loss, test_acc = saved_model.evaluate(testing_mel_spectrograms, testing_labels)
print('Test accuracy:', test_acc)