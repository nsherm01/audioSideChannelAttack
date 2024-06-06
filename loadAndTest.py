import os
from pytorchCoatnet import CoAtNet
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from keyLabel import keyLabel
import config

def main():
    # Load model weights
    model_weights = "model_weights.pth"
    testing_folder = "45_testing_output"

    total = 45
    total_correct = 0

    labeler = keyLabel()

    model = CoAtNet(config.img_size[0:2], config.img_size[2], config.num_blocks, config.channels, num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_weights))

    os.system('python3 librosaPeaks.py -test')

    for file in os.listdir(testing_folder):
        if file.endswith(".npy") and file[0] != "*":
            model.eval()
            x = np.load(os.path.join(testing_folder, file))
            x = cv2.resize(x, config.img_size[0, 2])
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            #print(x.shape)
            logits = model(x.float())
            #print(logits)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            predicted_class = labeler.inverse_transform([predicted_class.item()])[0]
            actual_class = file[0]
            if (predicted_class == actual_class):
                total_correct += 1

            print(f"Predicted Class: {predicted_class} vs. Actual Class: {actual_class}")
    print("Accuracy: ", total_correct/total)


if __name__ == '__main__':
    main()