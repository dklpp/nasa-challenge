import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV

# ---------------------------
# 0) Utils
# ---------------------------
def set_seed(seed=42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# 1) Load data (CSV)
# ---------------------------
# Replace with your paths
train_csv = "/home/senad/DataSet/exoplanet/exoTrain.csv"
test_csv  = "/home/senad/DataSet/exoplanet/exoTest.csv"

df_train = pd.read_csv(train_csv)
df_test  = pd.read_csv(test_csv)

# Many exoplanet CSVs have label in the first column (e.g., named 'LABEL' or index 0).
# We’ll robustly extract:
label_col = df_train.columns[0]
y_train_raw = df_train[label_col].values
y_test_raw  = df_test[label_col].values

# Convert labels to {0,1}
# If labels are already 0/1 this is a no-op; if not (e.g., 1/2) this maps to 0/1.
y_train = ((y_train_raw - y_train_raw.min()) / (y_train_raw.max() - y_train_raw.min())).astype(int)
y_test  = ((y_test_raw  - y_test_raw.min())  / (y_test_raw.max()  - y_test_raw.min())).astype(int)

# Drop the label column to get the time-series features
X_train = df_train.drop(columns=[label_col]).values
X_test  = df_test.drop(columns=[label_col]).values

# Optional: shuffle (your original used permutation)
perm_tr = np.random.permutation(len(X_train))
perm_te = np.random.permutation(len(X_test))
X_train, y_train = X_train[perm_tr], y_train[perm_tr]
X_test,  y_test  = X_test[perm_te],  y_test[perm_te]

# ---------------------------
# 2) Normalize rows (L2) and Gaussian smooth like before
# ---------------------------
X_train_norm = normalize(X_train)  # row-wise L2
X_test_norm  = normalize(X_test)

# Gaussian smoothing per row (equivalent to your gaussian_filter on axis=1)
sigma = 7.0
X_train_smooth = np.apply_along_axis(lambda r: gaussian_filter1d(r, sigma=sigma), 1, X_train_norm)
X_test_smooth  = np.apply_along_axis(lambda r: gaussian_filter1d(r, sigma=sigma), 1, X_test_norm)

# ---------------------------
# 3) FFT magnitude as features (same as your scipy.fft.fft2(abs))
# ---------------------------
# Your code used fft2 with axes=1; since data are 1D series, fft along axis=1 is sufficient:
X_train_fft = np.abs(scipy.fft.fft(X_train_smooth, axis=1))
X_test_fft  = np.abs(scipy.fft.fft(X_test_smooth, axis=1))

len_seq = X_train_fft.shape[1]

# ---------------------------
# 4) Oversampling to balance
# ---------------------------
ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)  # keep at least 1:2 minority:majority
X_train_bal, y_train_bal = ros.fit_resample(X_train_fft, y_train)
print("After oversampling -> class 1:", np.sum(y_train_bal == 1), "class 0:", np.sum(y_train_bal == 0))

# ---------------------------
# 5) PyTorch Dataset/Dataloader
# ---------------------------
class ExoDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, L) -> we’ll add channel dim on __getitem__
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]             # shape (L,)
        x = x[None, :]              # shape (1, L) for Conv1d
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor([y])

train_ds = ExoDataset(X_train_bal, y_train_bal)
test_ds  = ExoDataset(X_test_fft,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

# ---------------------------
# 6) FCN model (PyTorch) — mirrors your Keras architecture
# ---------------------------
class FCNModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=8)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn1   = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 340, kernel_size=6)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn2   = nn.BatchNorm1d(340)

        self.conv3 = nn.Conv1d(340, 256, kernel_size=4)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn3   = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)

        # figure out flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            h = self.pool1(F.relu(self.conv1(dummy)))
            h = self.bn1(h)
            h = self.pool2(F.relu(self.conv2(h)))
            h = self.bn2(h)
            h = self.pool3(F.relu(self.conv3(h)))
            h = self.bn3(h)
            flat = h.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 8)
        self.fc4 = nn.Linear(8, 1)  # BCEWithLogits

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)       # no sigmoid here
        return logits

model = FCNModel(len_seq).to(device)
print(model)

# ---------------------------
# 7) Train
# ---------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 15

train_losses = []
train_accs = []
val_accs = []

def evaluate_accuracy(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            y_true.append(yb.cpu().numpy())
            y_prob.append(prob.cpu().numpy())
    y_true = np.vstack(y_true).ravel()
    y_prob = np.vstack(y_prob).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return accuracy_score(y_true, y_pred)

for ep in range(1, epochs+1):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            running_correct += (pred.eq(yb)).sum().item()
            total += yb.numel()

    train_loss = running_loss / total
    train_acc = running_correct / total
    val_acc = evaluate_accuracy(test_loader)
    train_losses.append(train_loss); train_accs.append(train_acc); val_accs.append(val_acc)
    print(f"Epoch {ep:02d} | loss {train_loss:.4f} | acc {train_acc:.4f} | val_acc {val_acc:.4f}")

# ---------------------------
# 8) Curves
# ---------------------------
plt.figure(); plt.plot(train_accs, label="train_acc"); plt.plot(val_accs, label="val_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(True); plt.show()

plt.figure(); plt.plot(train_losses, label="train_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend(); plt.grid(True); plt.show()

# ---------------------------
# 9) Evaluate on test set (metrics + confusion matrix)
# ---------------------------
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
        y_prob.append(prob)
        y_true.append(yb.numpy().ravel())
y_true = np.concatenate(y_true)
y_prob = np.concatenate(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

print("Test accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))

cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No","Yes"])
disp.plot(cmap="Blues"); plt.title("PyTorch FCN Confusion Matrix"); plt.show()

# ---------------------------
# 10) SVC baseline on the same FFT features
# ---------------------------
params = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]
svc = GridSearchCV(svm.SVC(), param_grid=params, scoring='recall', n_jobs=-1)
svc.fit(X_train_bal, y_train_bal)
y_svc = svc.predict(X_test_fft)

print("\nSVC baseline:")
print(classification_report(y_test, y_svc, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))
cm2 = confusion_matrix(y_test.astype(int), y_svc.astype(int))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["No","Yes"])
disp2.plot(cmap="Blues"); plt.title("SVC Confusion Matrix"); plt.show()
