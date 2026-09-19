import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=1, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            # For demonstration, limit the training batches to speed up execution
            if i >= 10: 
                break
            
        # Validation
        model.eval()
        val_loss = []
        val_mae = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                mae = torch.mean(torch.abs(outputs - batch_y))
                
                val_loss.append(loss.item())
                val_mae.append(mae.item())
                
                if i >= 10:
                    break
                    
        print(f"Epoch: {epoch + 1} | Train Loss (MSE): {np.mean(train_loss):.4f} | Val Loss (MSE): {np.mean(val_loss):.4f} | Val MAE: {np.mean(val_mae):.4f}")
        
    return model

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = []
    test_mae = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            mae = torch.mean(torch.abs(outputs - batch_y))
            
            test_loss.append(loss.item())
            test_mae.append(mae.item())
            
            if i >= 10:
                break
                
    final_mse = np.mean(test_loss)
    final_mae = np.mean(test_mae)
    
    print(f"Final Test Metrics -> MSE: {final_mse:.4f} | MAE: {final_mae:.4f}")
    return final_mse, final_mae
