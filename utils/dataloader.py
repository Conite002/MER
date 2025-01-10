import json
import torch
from torch.utils.data import DataLoader, TensorDataset

import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def create_data_loaders(train_path, val_path, dims, batch_size=32):
    """
    Create DataLoaders for training and validation with dynamic dimensions.

    Args:
        train_path (str): Path to training data JSON.
        val_path (str): Path to validation data JSON.
        dims (dict): Dictionary specifying dimensions for audio, text, and video.
                     Example: {"audio": 100, "text": 768, "video": 512}
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: Training and validation DataLoaders for each modality.
        dict: Label mapping for class indices.
    """
    def load_data(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        audio_paths, video_paths, text_data, labels = [], [], [], []

        for item in data:
            audio_paths.append(item["audio"]) 
            video_paths.append(item["video"]) 
            text_data.append(item["text"])   
            labels.append(item["label"])     

        return audio_paths, video_paths, text_data, labels

    train_audio, train_video, train_text, train_labels = load_data(train_path)
    val_audio, val_video, val_text, val_labels = load_data(val_path)

    label_mapping = {label: idx for idx, label in enumerate(set(train_labels + val_labels))}
    train_labels = torch.tensor([label_mapping[label] for label in train_labels], dtype=torch.long)
    val_labels = torch.tensor([label_mapping[label] for label in val_labels], dtype=torch.long)

    train_audio_tensors = torch.zeros((len(train_audio), dims["audio"]), dtype=torch.float32)
    train_video_tensors = torch.zeros((len(train_video), dims["video"]), dtype=torch.float32)
    train_text_tensors = torch.zeros((len(train_text), dims["text"]), dtype=torch.float32)

    val_audio_tensors = torch.zeros((len(val_audio), dims["audio"]), dtype=torch.float32)
    val_video_tensors = torch.zeros((len(val_video), dims["video"]), dtype=torch.float32)
    val_text_tensors = torch.zeros((len(val_text), dims["text"]), dtype=torch.float32)

    train_loaders = {
        "audio": DataLoader(TensorDataset(train_audio_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "video": DataLoader(TensorDataset(train_video_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "text": DataLoader(TensorDataset(train_text_tensors, train_labels), batch_size=batch_size, shuffle=True),
    }
    val_loaders = {
        "audio": DataLoader(TensorDataset(val_audio_tensors, val_labels), batch_size=batch_size),
        "video": DataLoader(TensorDataset(val_video_tensors, val_labels), batch_size=batch_size),
        "text": DataLoader(TensorDataset(val_text_tensors, val_labels), batch_size=batch_size),
    }

    return train_loaders, val_loaders, label_mapping
