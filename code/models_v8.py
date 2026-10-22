"""v8: v6(SubjectAwareCNNLSTMAttention)와 동일 아키텍처, n_channels만 24(raw 12 + envelope 12)로 확장."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights


class SubjectAwareCNNLSTMAttention(nn.Module):
    def __init__(self, n_channels=24, n_classes=50, n_subjects=40, emb_dim=32, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.subject_emb = nn.Embedding(n_subjects + 1, emb_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(128 + emb_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden * 2)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(lstm_hidden * 2, n_classes)
        )

    def forward(self, x, subject_idx):
        s_emb = self.subject_emb(subject_idx)

        h = self.cnn(x)
        h = h.transpose(1, 2)

        T_prime = h.shape[1]
        s_emb_expanded = s_emb.unsqueeze(1).expand(-1, T_prime, -1)
        h_concat = torch.cat([h, s_emb_expanded], dim=-1)

        out, _ = self.lstm(h_concat)
        context, _ = self.attention(out)

        return self.fc(context)
