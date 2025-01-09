import torch

def evaluate_model(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        device (str): Device to evaluate on ('cpu' or 'cuda').

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy
