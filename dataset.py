import cv2
import numpy as np
from config import config
import torch
from torch.utils.data import Dataset

class DatasetMF(Dataset):
  def __init__(self, data, convert_to_rgb=False):
    self.data = data
    self.convert_to_rgb = convert_to_rgb

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.data[idx][0]
    y = self.data[idx][1]
    img = cv2.resize(np.asarray(x), config.img_size)
    label = np.asarray(y)

    print

    if self.convert_to_rgb:
      img = np.repeat(img[::, np.newaxis], 3, -1)

      return torch.tensor(img).view(3, config.img_size[0], config.img_size[1]), torch.tensor(label)

    else:

      return torch.tensor(img).view(3, config.img_size[0], config.img_size[1]), torch.tensor(label)