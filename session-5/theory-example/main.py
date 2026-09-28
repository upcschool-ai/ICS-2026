import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard
writer = SummaryWriter("runs/test")


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Create synthetic data: y = sum(x) + noise
n_samples = 5000
x = torch.randn(n_samples, 10)  # Input features
y_true = x.sum(dim=1, keepdim=True)  # Target is sum of features
y = y_true + 0.1 * torch.randn_like(y_true)  # Add some noise

# Split into train/test
train_size = int(0.8 * n_samples)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

for epoch in range(10):
    # Train on batches
    batch_size = 32
    total_train_loss = 0
    num_batches = 0
    
    for i in range(0, train_size, batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_batches += 1

    # Calculate average training loss
    avg_train_loss = total_train_loss / num_batches
    
    # Calculate test loss
    with torch.no_grad():
        test_output = model(x_test)
        test_loss = criterion(test_output, y_test)
    
    # Log losses
    writer.add_scalar("Train Loss", avg_train_loss, epoch)
    writer.add_scalar("Test Loss", test_loss.item(), epoch)
    
    # Log parameter histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f"{name}/grad", param.grad, epoch)
        
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss.item():.4f}")

writer.close()


print("What is wrong with this code..?")
