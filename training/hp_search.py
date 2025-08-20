import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

from data_utils import train_dataloader, valid_dataloader, test_dataloader
from model import NeuralNetwork
from train_utils import train, evaluate

space = {
    "num_hidden_layers": [1, 2, 3, 4, 5],
    "num_hidden_units": [16, 32, 64, 128, 256],
    "optim": [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
    "learning_rate": [0.1, 0.01, 0.001, 0.0001],
    "dropout": [0.0, 0.25, 0.5],
    "weight_decay": [10, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
}


n_trials = 10
results = []
X_batch, y_batch = next(iter(train_dataloader))


best_val = float('inf')
best_model_state = None
for trial in range(n_trials):
    print(f"Starting trial {trial+1}/{n_trials}")
    config = {k: random.choice(v) for k, v in space.items()}
    model = NeuralNetwork(
        input_size=X_batch.shape[1],
        hidden_dim=config["num_hidden_units"],
        num_layers=config["num_hidden_layers"],
        dropout_rate=config["dropout"]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = config["optim"](
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    train_losses, valid_losses = train(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        epochs=200,
    )
    print(f"Finished trial {trial+1}")
    current_best_val = min(valid_losses)
    if current_best_val < best_val:
        best_val = current_best_val
        best_model_state = model.state_dict()
    results.append((config, train_losses, valid_losses))

best_vals = []
best_epochs = []
for _, _, va in results:
    val = min(va)
    epoch = va.index(val) + 1
    best_vals.append(val)
    best_epochs.append(epoch)

best_idx = int(min(range(len(best_vals)), key=lambda i: best_vals[i]))

n_cols = 4
n_rows = (len(results) + n_cols - 1) // n_cols
plt.figure(figsize=(4 * n_cols, 3 * n_rows))

for idx, (config, tr, va) in enumerate(results):
    ax = plt.subplot(n_rows, n_cols, idx + 1)
    ax.plot(tr, label="train")
    ax.plot(va, label="valid")

    this_best = best_vals[idx]
    this_epoch = best_epochs[idx]

    title = (
        f"layers={config['num_hidden_layers']}, units={config['num_hidden_units']}\n"
        f"opt={config['optim'].__name__}, lr={config['learning_rate']}\n"
        f"drop={config['dropout']}, wd={config['weight_decay']}\n"
        f"best_val={this_best:.4f} @ epoch {this_epoch}"
    )

    if idx == best_idx:
        ax.set_facecolor((1.0, 0.0, 0.0, 0.2))

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize="small")

plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()

print("Length of results:", len(results))
print("Best index:", best_idx)
print("Type of results:", type(results))
print("First element type:", type(results[0]))
print("Accessing results[best_idx] now...")
best_config, _, _ = results[best_idx]
print("About to choose device")

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)    
best_model = NeuralNetwork(
    input_size=X_batch.shape[1],
    hidden_dim=best_config["num_hidden_units"],
    num_layers=best_config["num_hidden_layers"],
    dropout_rate=best_config["dropout"]
).to(device)
best_model.load_state_dict(best_model_state)
best_model.eval() 

criterion = nn.CrossEntropyLoss()

test_loss, test_accuracy = evaluate(best_model, test_dataloader, criterion, device)
print(f"\nBest model test loss: {test_loss:.4f}")
print(f"Best model test accuracy: {test_accuracy:.2%}")

# Saving the model
"""torch.save({
    'model_state_dict': best_model.state_dict(),
    'config': best_config
}, 'best_model.pth')"""
