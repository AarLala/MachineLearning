import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = Network()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def test():
    #was lazy to build an actual test set
    outputs = model(X)
    loss = criterion(outputs, y)
    return(loss)
X = torch.tensor([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1],
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
], dtype=torch.float32)


epochs = 20000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print(test())
