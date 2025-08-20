import numpy as np
import matplotlib.pyplot as plt

import torch

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
      for batch_X, batch_y in dataloader:
        batch_y = batch_y.squeeze().long()
        preds_y = model(batch_X)
        loss = criterion(preds_y, batch_y)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
      epoch_losses = []
      for batch_X, batch_y in train_dataloader:
        batch_y = batch_y.squeeze().long()
        preds_y = model(batch_X)
        loss = criterion(preds_y, batch_y)
        epoch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      train_loss = np.mean(epoch_losses)
      train_losses.append(train_loss)

      valid_loss = validate(model, valid_dataloader, criterion)
      valid_losses.append(valid_loss)
    return train_losses, valid_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze().long()
            preds_y = model(batch_X)
            loss = criterion(preds_y, batch_y)
            total_loss += loss.item() * batch_X.size(0)

            pred_labels = preds_y.argmax(dim=1)
            correct += (pred_labels == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    avg_accuracy = correct / total
    return avg_loss, avg_accuracy


def plot_losses(train_losses, valid_losses):
  plt.plot(train_losses, label='Training Loss')
  plt.plot(valid_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()