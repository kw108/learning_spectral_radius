import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


sampling_rate = '1e5'
fading_type = 'rayleigh'
network_type = 'ICNN'   # ICNN or DNN

# Load preprocessed channel gain matrices and eigentriples
with open('eigentriples_' + sampling_rate + '.pkl', 'rb') as file:
    rayleigh_gain = pickle.load(file)
    rayleigh_w = pickle.load(file)
    rayleigh_lv = pickle.load(file)
    rayleigh_rv = pickle.load(file)
    rician_gain = pickle.load(file)
    rician_w = pickle.load(file)
    rician_lv = pickle.load(file)
    rician_rv = pickle.load(file)
    if fading_type == 'rayleigh':
        gain, w, lv, rv = rayleigh_gain, rayleigh_w, rayleigh_lv, rayleigh_rv
    if fading_type == 'rician':
        gain, w, lv, rv = rician_gain, rician_w, rician_lv, rician_rv

# Create datasets for regression
X, y = np.log(gain / np.median(w)), np.log(w / np.median(w))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# Define the positive linear layer
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, network_type):
        super(PositiveLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.network_type = network_type

    def forward(self, x):
        if self.network_type == 'ICNN':
            weight = torch.clamp_min(self.weight, 0.0)
        else:
            weight = self.weight
        return nn.functional.linear(x, weight, self.bias)


# Define the input convex neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc11 = PositiveLinear(32, 8, network_type)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        z1 = torch.relu(self.fc1(x))
        z2 = torch.relu(self.fc11(z1) + self.fc2(x))
        z3 = self.fc3(z2)
        return z3


# Define the loss function and optimizer
class CustomCriterion(nn.Module):
    def __init__(self, weight=None):
        super(CustomCriterion, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.mean(torch.square(torch.exp(inputs) - torch.exp(targets)))
        return loss


# Create an instance of the neural network model
model = Net()
criterion = CustomCriterion()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the data loaders for training and validation
batch_size = 32
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Print model summary
print(summary(model, X_train.numpy().shape))

# Define the early stopping criteria
patience = 10
min_delta = 0.1
best_loss = float('inf')
counter = 0

# Train the model
for epoch in range(100):
    train_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            val_loss += loss.item()

    # Check for early stopping
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping after {epoch+1} epochs.')
            break

    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')
