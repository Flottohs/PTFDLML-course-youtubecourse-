import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------

# Create a circular dataset for binary classification
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Visualize the data structure in a DataFrame
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
print(f"Dataset Head:\n{circles.head()}")

# Visualize the circles
plt.figure(figsize=(6, 6))
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("Visualizing the Circle Data")
plt.show()

# Setup Device-Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Convert data to Tensors and move to target device immediately
X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

# Split into Training and Test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. MODEL BUILDING
# ---------------------------------------------------------

class CircleModelV1(nn.Module):
    """
    Simple Linear Neural Network for binary classification.
    Note: Without non-linear activation (ReLU), this model 
    will struggle to learn the circular boundary.
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10) 
        self.layer_2 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

model_0 = CircleModelV1().to(device)

# Alternative: Define the same architecture using nn.Sequential
model_seq = nn.Sequential(
    nn.Linear(2, 10),
    nn.Linear(10, 1)
).to(device)

# ---------------------------------------------------------
# 3. LOSS, OPTIMIZER, AND UTILITIES
# ---------------------------------------------------------

# BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy for better stability
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy percentage."""
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ---------------------------------------------------------
# 4. TRAINING AND EVALUATION LOOP
# ---------------------------------------------------------

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs + 1):
    # --- Training Mode ---
    model_0.train()
    
    # 1. Forward pass (outputs raw Logits)
    y_logits = model_0(X_train).squeeze()
    # 2. Convert Logits -> Probabilities -> Labels
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # 3. Calculate Loss and Accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    # 4. Optimizer Zero Grad
    optimizer.zero_grad()
    # 5. Backpropagation
    loss.backward()
    # 6. Step the Optimizer
    optimizer.step()
    
    # --- Evaluation Mode ---
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# ---------------------------------------------------------
# 5. VISUALIZATION OF RESULTS
# ---------------------------------------------------------

# Download helper functions if not existing
if not Path("helper_functions.py").is_file():
    print("Downloading helper_functions.py...")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_decision_boundary

# Plot Decision Boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Results")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test Results")
plot_decision_boundary(model_0, X_test, y_test)
plt.show() 




