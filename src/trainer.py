import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def _limited_batches(loader, max_batches=None):
    """Yield all batches, or an explicit subset for quick demonstrations."""
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def _evaluate_loader(model, data_loader, device, max_batches=None):
    """Return element-weighted MSE/MAE for a loader."""
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in _limited_batches(data_loader, max_batches):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)

            # Sum errors so a short final batch receives the correct weight.
            squared_error += torch.sum((outputs - batch_y) ** 2).item()
            absolute_error += torch.sum(torch.abs(outputs - batch_y)).item()
            element_count += batch_y.numel()

    if element_count == 0:
        raise ValueError("The data loader produced no samples.")

    return squared_error / element_count, absolute_error / element_count


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=1,
    learning_rate=0.001,
    device='cpu',
    max_batches=None,
):
    """
    Train on the complete loaders by default.

    ``max_batches`` is available only for explicitly requested quick demos.  It
    defaults to ``None`` so experiments and saved checkpoints use every batch.
    The state with the lowest validation MSE is restored before returning.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_val_mse = float('inf')
    best_state = None

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        train_squared_error = 0.0
        train_element_count = 0

        for batch_x, batch_y in _limited_batches(train_loader, max_batches):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_squared_error += torch.sum((outputs.detach() - batch_y) ** 2).item()
            train_element_count += batch_y.numel()

        if train_element_count == 0:
            raise ValueError("The training loader produced no samples.")

        train_mse = train_squared_error / train_element_count
        val_mse, val_mae = _evaluate_loader(
            model, val_loader, device, max_batches=max_batches
        )
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch: {epoch + 1} | Train Loss (MSE): {train_mse:.4f} | "
            f"Val Loss (MSE): {val_mse:.4f} | Val MAE: {val_mae:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history = history
    return model


def evaluate_model(model, test_loader, device='cpu', max_batches=None):
    """Evaluate every test sample unless ``max_batches`` is explicitly set."""
    model.to(device)
    final_mse, final_mae = _evaluate_loader(
        model, test_loader, device, max_batches=max_batches
    )
    print(f"Final Test Metrics -> MSE: {final_mse:.4f} | MAE: {final_mae:.4f}")
    return final_mse, final_mae
