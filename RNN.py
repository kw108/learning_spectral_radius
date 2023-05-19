import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import pickle


sampling_rate = '1e4'
fading_type = 'rician'
uniform = False

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

# generate some data
np.random.seed(0)
num_samples = 100
seq_len = 10
input_dim = 64
hidden_dim = 29
output_dim = 1

if uniform:
    # use sampling at non-uniform intervals
    indices = list(range(0, 100000, 10))
else:
    # use sampling at non-uniform intervals
    indices = sorted(np.random.choice(list(range(100000)), 10000, replace=False))
gain = gain[indices]
w = w[indices]
lv = lv[indices]
rv = rv[indices]

m = np.median(w)
rayleigh_gain = gain / m
rayleigh_w = w / m
x = np.zeros((num_samples, seq_len, input_dim))
y = np.zeros((num_samples, 1))
for i in range(num_samples):
    x[i, :, :] = rayleigh_gain[i * seq_len: (i + 1) * seq_len, :]
    y[i, 0] = rayleigh_w[i * seq_len - 1]


class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])
        return out


# define the model, loss function, and optimizer
model = Seq2SeqRNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model with early stopping
best_loss = np.inf
patience = 100
for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    inputs = torch.tensor(x, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # evaluate the model on a validation set
    model.eval()
    with torch.no_grad():
        inputs_val = torch.tensor(x, dtype=torch.float32)
        targets_val = torch.tensor(y, dtype=torch.float32)
        outputs_val = model(inputs_val)
        val_loss = criterion(outputs_val, targets_val)

    # check if validation loss has improved
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        print(f"training loss {loss} with validation loss {val_loss} at epoch {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch} with validation loss {val_loss}")
            break
