import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, n_channels=12, n_classes=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ELU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        return self.fc(self.net(x))


class CNNLSTM(nn.Module):
    def __init__(self, n_channels=12, n_classes=50, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(lstm_hidden * 2, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        h = self.cnn(x)               # (B, 128, T')
        h = h.transpose(1, 2)         # (B, T', 128)
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.fc(last)
