import torch
import torch.nn as nn
import torchvision
from skorch import NeuralNetClassifier
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import glob
import os
import pandas as pd
from tqdm import tqdm

# Define your ResNet class
class ResNet(nn.Module):
    def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.n_classes = n_classes
        models_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }
        self.base_model = models_dict[model](weights=None)
        self._feature_vector_dimension = self.base_model.fc.in_features
        self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
        self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        x = self.fc(features)
        return x

def get_scores(net, X_test, batch_size=64):
    # Ensure the model is in evaluation mode
    net.module_.eval()

    # Store all scores
    all_scores = []

    # Convert X_test to torch tensor if it's a numpy array
    if isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).float()

    # Process in batches
    with torch.no_grad():
        for start_idx in range(0, len(X_test), batch_size):
            end_idx = start_idx + batch_size
            batch_x = X_test[start_idx:end_idx]

            # Skorch handles the device, so we directly forward the data to the model
            scores = net.forward(batch_x)
            probabilities = torch.softmax(scores, dim=1)
            all_scores.append(probabilities.cpu())

    # Concatenate all batch scores
    all_scores = torch.cat(all_scores, dim=0)

    return all_scores.numpy()

# Parameters
model = 'resnet34'
batch_size = 4096*2
model_filename = './resnet34_4.pt'

# Initialize the Skorch wrapper
net = NeuralNetClassifier(
    module=ResNet,
    module__model=model,  # Choose the ResNet model you want
    module__n_channels=4,
    batch_size=batch_size,
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

# Initialize the model
net.initialize()

# Load the saved model parameters
net.load_params(f_params=model_filename)

# Annotations def
annotation_dict = {'non-parasite': 0, 'parasite': 1, 'unlabeled': -1}

# Go though the npy files
data_dir = '/home/octopi/Desktop/Octopi/data/npy_v2'
output_dir = '../model output'

npy_files = glob.glob(os.path.join(data_dir, '*.npy'))

for file_path in tqdm(npy_files):

    print(file_path)
    
    X = np.load(file_path)
    X = X.astype(np.float32) / 255.0

    # Start the timer
    start_time = time.time()
    
    # Run inference
    scores = get_scores(net, X, batch_size)

    # End the timer
    end_time = time.time()

    # Calculate total time and average time per image
    total_time = end_time - start_time
    num_images = len(X)
    avg_time_per_image = total_time / num_images

    # Print the time taken, number of images, and time per image
    print(f"File: {os.path.basename(file_path)} - Total Time: {total_time:.2f} seconds, Number of Images: {num_images}, Average Time per Image: {avg_time_per_image:.4f} seconds")

    # Create a DataFrame for the result
    if scores.shape[0] != X.shape[0]:
        raise ValueError("Mismatch between number of predictions and number of images")
    output_df = pd.DataFrame(scores, columns=[key + ' output' for key in annotation_dict if annotation_dict[key] >= 0])

    # Sort the DataFrame by 'parasite output' in descending order
    output_df = output_df.sort_values(by='parasite output', ascending=False)

    # Save the DataFrame as a CSV file
    csv_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
    output_df.to_csv(csv_file_path, index=True, index_label='index')