from sys import argv
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from pytorchCoatnet import coatnet_0
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from melDataset import melDataset

batch_size = 8
num_splits = 4
epochs = 10
lr = 5e-5 # taken from the paper
weight_decay = 1e-8 # taken from the paper

model = coatnet_0()

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

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
    

    return list(zip(mel_spectrograms, labels))

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
    return list(zip(testing_mel_spectrograms, testing_labels))
    
def train_epoch(model, data_loader):

    model.train()
    train_loss, train_correct = 0.0, 0

    for step, batch in enumerate(data_loader):
        optim.zero_grad()
        x, y = batch
        x = x.unsqueeze(1)
        print("Training step: ", step)
        logits = model(x.float())
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        loss.backward()
        optim.step()

        # calculate accuracy
        preds = torch.argmax(logits, dim=1).flatten()
        correct_preds_n = (preds == y).cpu().sum().item()
        train_correct += correct_preds_n
    return train_loss, train_correct

def valid_epoch(model, train_dataloader):
  model.eval()
  val_loss, val_correct = 0.0, 0
  
  for step, batch in enumerate(train_dataloader):
    optim.zero_grad()
    x, y = batch

    logits = model(x.float()) 
    loss = loss_fn(logits, y)
    val_loss += loss.item()
    loss.backward()
    optim.step()
    preds = torch.argmax(logits, dim=1).flatten()
    correct_preds_n = (preds == y).cpu().sum().item()
    val_correct += correct_preds_n
  
  return val_loss, val_correct

def main():
    if (len(argv) > 1):
        training_data_directory = argv[1]
    else:
        training_data_directory = 'mel_spectrograms_(128x321)'

    fold_history = {}

    training_data = load_training_data(training_data_directory)
    testing_data = load_testing_data('testing_output')

    dataset = melDataset(training_data)
    # training_data = (np_array, int)
    # dataset = (tensor, tensor)

    splits=KFold(n_splits=num_splits, shuffle=True, random_state=1337)

    for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    
        for epoch in range(epochs):
            torch.cuda.empty_cache()
            print('---train:')    
            train_loss, train_correct = train_epoch(model, train_loader)
            print('---eval:')
            test_loss, test_correct = valid_epoch(model, test_loader)
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            print('---status:')
            print("\tEpoch:{}/{} \n\tAverage Training Loss:{:.4f}, Average Test Loss:{:.4f}; \n\tAverage Training Acc {:.2f}%, Average Test Acc {:.2f}%\n".format(epoch + 1,
                                                                                                                                                                config.epochs,
                                                                                                                                                                train_loss,
                                                                                                                                                                test_loss,
                                                                                                                                                                train_acc,
                                                                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
        
        fold_history[f'fold{fold+1}'] = history

if __name__ == "__main__":
    main()