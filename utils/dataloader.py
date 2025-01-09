import json
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_data_loaders(train_path, val_path, batch_size=32):
    """
    Create DataLoaders for training and validation.

    Args:
        train_path (str): Path to training data JSON.
        val_path (str): Path to validation data JSON.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: Training and validation DataLoaders for each modality.
    """
    def load_data(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        audio, text, video, labels = [], [], [], []
        for item in data:
            audio.append(item["audio"])
            text.append(item["text"])
            video.append(item["visual"])
            labels.append(item["labels"])
        return torch.tensor(audio), torch.tensor(text), torch.tensor(video), torch.tensor(labels)

    train_audio, train_text, train_video, train_labels = load_data(train_path)
    val_audio, val_text, val_video, val_labels = load_data(val_path)

    train_loaders = {
        "audio": DataLoader(TensorDataset(train_audio, train_labels), batch_size=batch_size, shuffle=True),
        "text": DataLoader(TensorDataset(train_text, train_labels), batch_size=batch_size, shuffle=True),
        "video": DataLoader(TensorDataset(train_video, train_labels), batch_size=batch_size, shuffle=True),
    }
    val_loaders = {
        "audio": DataLoader(TensorDataset(val_audio, val_labels), batch_size=batch_size),
        "text": DataLoader(TensorDataset(val_text, val_labels), batch_size=batch_size),
        "video": DataLoader(TensorDataset(val_video, val_labels), batch_size=batch_size),
    }

    return train_loaders, val_loaders
