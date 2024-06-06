import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

'''
melDataset.py

This file contains the melDataset class, which is used to create a PyTorch Dataset object for the mel spectrograms and their corresponding labels.
'''
class melDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.data[idx][0]
    y = self.data[idx][1]

    return torch.tensor(cv2.resize(x, (224, 224))), torch.tensor(y)