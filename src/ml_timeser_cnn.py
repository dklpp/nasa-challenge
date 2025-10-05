import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from imblearn.over_sampling import RandomOverSampler
from scipy.ndimage import gaussian_filter1d
import scipy.fft

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
from models import FCNModel  # your models.py file (BatchNorm can be replaced below)

DATA_PATH = "data/processed_tess/tess_data_2000_filtered.csv"
OUTPUT_DIR = "outputs_CNN_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-4             # reduced LR for stability
SIGMA = 7.0
SEED = 42

# ------------------------------------------------------------
# 2. Utils
# ------------------------------------------------------------
def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)

# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape, "columns")

# Split features and labels
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].astype(np.float32).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("SANITY CHECKS:")
print("Unique train labels:", np.unique(y_train, return_counts=True))
print("Unique test  labels:", np.unique(y_test,  return_counts=True))


# ------------------------------------------------------------
# 4. Normalize + smooth + FFT (with log scaling + standardization)
# ------------------------------------------------------------
# Row-wise L2 normalization
X_train = normalize(X_train)
X_test = normalize(X_test)

# Gaussian smoothing per sample
X_train = np.apply_along_axis(lambda r: gaussian_filter1d(r, sigma=SIGMA), 1, X_train)
X_test = np.apply_along_axis(lambda r: gaussian_filter1d(r, sigma=SIGMA), 1, X_test)

# FFT magnitude → log(1 + abs(FFT))
X_train_fft = np.log1p(np.abs(scipy.fft.fft(X_train, axis=1)))
X_test_fft  = np.log1p(np.abs(scipy.fft.fft(X_test, axis=1)))

# Standardize frequency-domain features
scaler = StandardScaler()
X_train_fft = scaler.fit_transform(X_train_fft)
X_test_fft  = scaler.transform(X_test_fft)

print("FFT shape:", X_train_fft.shape)
print("Train mean/std:", np.mean(X_train_fft), np.std(X_train_fft))

# ------------------------------------------------------------
# 5. Balance training data
# ------------------------------------------------------------
ros = RandomOverSampler(random_state=SEED)
X_train_bal, y_train_bal = ros.fit_resample(X_train_fft, y_train)
print(f"After balancing -> class0={np.sum(y_train_bal==0)}, class1={np.sum(y_train_bal==1)}")
print("Unique vals:", np.unique(y_train_bal), np.unique(y_test))

# ------------------------------------------------------------
# 6. Dataset / Dataloader
# ------------------------------------------------------------
class TessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx][None, :]  # (1, seq_len)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


train_ds = TessDataset(X_train_bal, y_train_bal)
test_ds  = TessDataset(X_test_fft, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# ------------------------------------------------------------
# 7. Model, loss, optimizer
# ------------------------------------------------------------
seq_len = X_train_fft.shape[1]
model = FCNModel(seq_len).to(device)

# Disable BatchNorm if dataset small → replace with Identity
for name, module in model.named_children():
    if isinstance(module, nn.BatchNorm1d):
        setattr(model, name, nn.Identity())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)

# ------------------------------------------------------------
# 8. Training loop
# ------------------------------------------------------------
def evaluate(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy().ravel())
            y_prob.extend(prob.cpu().numpy().ravel())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return accuracy_score(y_true, y_pred)


train_losses, train_accs, val_accs = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds.eq(yb)).sum().item()
            total += yb.numel()

    train_loss = total_loss / total
    train_acc = correct / total
    val_acc = evaluate(test_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch:02d}: loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

# ------------------------------------------------------------
# 9. Curves
# ------------------------------------------------------------
plt.figure()
plt.plot(train_accs, label="train_acc")
plt.plot(val_accs, label="val_acc")
plt.legend(); plt.title("Accuracy"); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.show()

plt.figure()
plt.plot(train_losses, label="loss")
plt.legend(); plt.title("Loss"); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
plt.show()

# ------------------------------------------------------------
# 10. Evaluation on test set
# ------------------------------------------------------------
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
        y_prob.extend(prob)
        y_true.extend(yb.numpy().ravel())

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

print("\n✅ Test Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["class 0 (FP/FA)", "class 1 (CP/KP)"]))

cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["class 0", "class 1"])
disp.plot(cmap="Blues")
plt.title("FCN Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()
