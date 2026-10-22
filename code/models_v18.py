"""
v18: DANN (Domain Adversarial Neural Network) for subject-invariant EMG classification.

Feature extractor: CNN + BiLSTM + attention → 256-dim features (no subject embedding).
Label head:   Dropout(0.4) + Linear(256, n_classes)
Domain head:  GRL(alpha) + Linear(256, 128) + ReLU + Dropout(0.3) + Linear(128, n_subjects)

The GRL reverses gradient sign (scaled by alpha) during backprop, forcing the feature
extractor to produce features that are maximally uninformative about subject identity
while still discriminating gestures.
"""
import torch
import torch.nn as nn


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DANNEmgClassifier(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_subjects: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            128, 128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.attn = nn.Linear(256, 1)

        self.label_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_subjects),
        )

    def _features(self, x):
        h = self.conv(x)               # (B, 128, T)
        h = h.transpose(1, 2)          # (B, T, 128)
        h, _ = self.lstm(h)            # (B, T, 256)
        a = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        return (h * a).sum(dim=1)      # (B, 256)

    def forward(self, x, alpha: float = 0.0):
        feat = self._features(x)
        class_logits = self.label_head(feat)
        domain_logits = self.domain_head(_GRL.apply(feat, alpha))
        return class_logits, domain_logits
