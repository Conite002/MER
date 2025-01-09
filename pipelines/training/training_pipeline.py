import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pipelines.evaluation.evaluation_pipeline import evaluate_model

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Train a single model.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        nn.Module: Trained model.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

    return model



