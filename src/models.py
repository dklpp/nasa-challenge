import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# https://github.com/senad96/Exoplanet-Detection/blob/main/project_main.py

class FCNModel(nn.Module):
    def __init__(self, len_seq):
        super(FCNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=8)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=340, kernel_size=6)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn2 = nn.BatchNorm1d(340)

        self.conv3 = nn.Conv1d(in_channels=340, out_channels=256, kernel_size=4)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Dense (fully connected) layers
        self.fc1 = nn.Linear(self._get_flatten_size(len_seq), 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 8)
        self.fc4 = nn.Linear(8, 1)

    def _get_flatten_size(self, len_seq):
        """Run dummy input through conv/pool layers to find flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, len_seq)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            return x.numel()

    def forward(self, x):
        # x shape: (batch, 1, len_seq)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.bn3(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x

def SVC_model():
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
         'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]
    clf = GridSearchCV(SVC(), param_grid=tuned_parameters, scoring='recall')
    return clf

# # Example input sequence length
# len_seq = 2000

# model = FCNModel(len_seq)
# print(model)

# # Example input batch (batch_size=8)
# x = torch.randn(8, 1, len_seq)
# y_pred = model(x)
# print("Output shape:", y_pred.shape)

# import torch.optim as optim

# criterion = nn.BCELoss()  # for binary classification
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(10):
#     optimizer.zero_grad()
#     outputs = model(x)
#     labels = torch.randint(0, 2, (8, 1), dtype=torch.float32)  # dummy labels
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

