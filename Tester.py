# Title: Tester.py
# Authors: Rem D'Ambrosio
# Created: 2024-12-02
# Description: initial structure by Mayur Ingole

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

from CNN import CNN


class Tester(): 
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def test_model(self, model):
        """
        Tests trained model
        """
        batch_size = 64

        test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        num_classes = 10
        num_correct = 0
        num_samples = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                scores = model(x)
                _, predictions = scores.max(1)

                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)

                for class_label in range(num_classes):
                    true_positives += ((predictions == class_label) & (y == class_label)).sum().item()
                    true_negatives += ((predictions != class_label) & (y != class_label)).sum().item()
                    false_positives += ((predictions == class_label) & (y != class_label)).sum().item()
                    false_negatives += ((predictions != class_label) & (y == class_label)).sum().item()

            accuracy = float(num_correct) / float(num_samples)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return


    def demo_model(self, model, sample_path):
        """
        Demonstrates model on single file
        """
        image = self.load_bmp(sample_path)

        with torch.no_grad():
            scores = model(image)
            _, prediction = scores.max(1)

        print(f"Letter in {sample_path}: {prediction.item()}")

        return


# ==================================================================================================
# I/O FUNCTIONS
# ==================================================================================================


    def load_model(self, model_path):
        """
        Loads trained model from file
        """
        model = CNN()                                     # create empty model
        model.load_state_dict(torch.load(model_path))     # load static dict with trained params
        model.eval()                                      # set model to evaluation mode
        return model
 

    def load_bmp(self, bmp_path):
        """
        Loads image from bmp file and converts to tensor
        """
        with open(bmp_path, 'rb') as f:
            f.seek(54)
            width, height = 28, 28
            row_size = (width * 3 + 3) & ~3  # row_size is padded to multiple of 4 bytes (RGB)

            pixels = []
            for y in range(height):
                row_data = f.read(row_size)
                row_rgb = list(row_data[:width * 3])
                row_rgb = [row_rgb[i:i+3] for i in range(0, len(row_rgb), 3)]
                pixels.insert(0, row_rgb)

            pixel_array = np.array(pixels, dtype=np.uint8)
            grayscale_image = np.mean(pixel_array, axis=-1).astype(np.uint8)
            image_tensor = torch.tensor(grayscale_image, dtype=torch.float32) / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        return image_tensor