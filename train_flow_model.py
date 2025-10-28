"""
Train a simple flow-matching model on Lorenz-63 trajectories.

FM objective (straight-path):
  v_theta(x_s, s) ≈ x1 - x0
where x_s = (1 - s) * x0 + s * x1,  s ~ Uniform[0,1]
"""

import os
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import MLP




path = "/Users/mayankshrivastava/Desktop/DataAssimilation/FlowDAS/experiments/lorenz/data/dataset/train.h5"
with h5py.File(path, "r") as f:
    print("Keys:", list(f.keys()))
    X = f["data"][:]   


X = torch.from_numpy(X).float()         
N, T, D = X.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)



net = MLP(x_dim=D, o_dim=0, hidden=128).to(device)


lr = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()   # between predicted drift and (x1 - x0)


epochs = 20                  
losses = []

net.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    
    for t_idx in range(T - 1):
        optimizer.zero_grad()

        
        x0 = X[:, t_idx, :]       # [N, 3]
        x1 = X[:, t_idx + 1, :]   # [N, 3]

        
        s = torch.rand(N, 1, device=device)

        
        x_s = (1.0 - s) * x0 + s * x1    # [N, 3]

        
        v_target = x1 - x0              

        
        v_pred = net(x_s, s.squeeze(-1), o_t=None)  

        loss = criterion(v_pred, v_target)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= (T - 1)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}  |  loss={epoch_loss:.6f}")
    

import datetime, torch

date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"train_model_{date}.pt"

torch.save(net.state_dict(), save_path)
print(f"Model saved to {save_path}")


plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("FM Loss (MSE)")
plt.title("Flow Matching Training Loss (Lorenz-63)")
plt.tight_layout()
plt.show()
