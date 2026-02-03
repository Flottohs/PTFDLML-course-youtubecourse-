import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from helper_functions import plot_decision_boundary


# -----------------------------
# utility
# -----------------------------
def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).float().mean() * 100


# -----------------------------
# device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# data
# -----------------------------
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1,
    random_state=RANDOM_SEED
)

X_blob = torch.tensor(X_blob, dtype=torch.float32)
y_blob = torch.tensor(y_blob, dtype=torch.long)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

X_blob_train = X_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_train = y_blob_train.to(device)
y_blob_test = y_blob_test.to(device)


# visualize dataset
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.title("Blob Dataset")
plt.show()


# -----------------------------
# model
# -----------------------------
class MODEL_AWESOME(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.net(x)


model = MODEL_AWESOME(
    input_features=NUM_FEATURES,
    output_features=NUM_CLASSES,
    hidden_units=128
).to(device)


# -----------------------------
# training setup
# -----------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# -----------------------------
# training loop
# -----------------------------
epochs = 1000
for epoch in range(epochs):
    # ---- train ----
    model.train()
    train_logits = model(X_blob_train)
    train_preds = train_logits.argmax(dim=1)

    loss = loss_fn(train_logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, train_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- test ----
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_blob_test)
        test_preds = test_logits.argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_preds)

    if epoch % 100 == 0:
        print(
            f"epoch {epoch:4d} | "
            f"train acc {acc:6.2f}% | train loss {loss:.4f} | "
            f"test acc {test_acc:6.2f}% | test loss {test_loss:.4f}"
        )


# -----------------------------
# decision boundaries
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_blob_test, y_blob_test)

plt.show()
