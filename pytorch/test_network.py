import torch
import torch.nn as nn
from Network import Network

inputs = torch.rand((32, 5220))  # 32 samples, each of 29 features
labels = torch.randint(0, 2, (32,2)).float()  # 32 labels (either 0 or 1)

model = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

# Training loop (example for one epoch)
model.train()
optimizer.zero_grad()

outputs = model(inputs).squeeze()  # Forward pass
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    sample_input = torch.rand((1, 5220))
    predicted_probabilities = model(sample_input)
    predicted_label = (predicted_probabilities > 0.5).float()  # Use threshold of 0.5 to get binary label

print(predicted_label)





