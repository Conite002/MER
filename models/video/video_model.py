import torch
import torch.nn as nn


class VideoMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        MLP-based classifier for video embeddings.
        Parameters:
        - input_dim: int, number of input features (e.g., 768 for ViT embeddings).
        - num_classes: int, number of output classes.
        """
        super(VideoMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
