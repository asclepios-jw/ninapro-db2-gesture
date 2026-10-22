"""v11: v8와 동일 아키텍처(SubjectAwareCNNLSTMAttention)에 Center Loss를 적용하기 위해
forward()가 logits와 attention context 특징(분류기 직전 벡터)을 함께 반환하도록 확장."""
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
        self.feature_dim = lstm_hidden * 2

    def forward(self, x, subject_idx, return_features=False):
        s_emb = self.subject_emb(subject_idx)

        h = self.cnn(x)
        h = h.transpose(1, 2)

        T_prime = h.shape[1]
        s_emb_expanded = s_emb.unsqueeze(1).expand(-1, T_prime, -1)
        h_concat = torch.cat([h, s_emb_expanded], dim=-1)

        out, _ = self.lstm(h_concat)
        context, _ = self.attention(out)

        logits = self.fc(context)
        if return_features:
            return logits, context
        return logits


class CenterLoss(nn.Module):
    """클래스별 학습 가능한 center를 두고, 같은 클래스의 feature가 그 center에
    가까워지도록 당기는 손실 (intra-class compactness). FocalLoss/class weight와는
    무관한 별개 메커니즘 — v3~v5에서 문제였던 "쉬운 클래스 손실을 깎는" 효과가 없어
    rest 클래스를 무시하게 만들 위험이 없다."""

    def __init__(self, n_classes: int, feature_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_classes, feature_dim) * 0.1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()
