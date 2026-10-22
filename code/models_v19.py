"""
v19: DANN with subject embedding preserved.

Feature extractor (CNN + BiLSTM + attention) → 256-dim features
Domain head: GRL(alpha) + MLP → 2 classes (hard=0 / easy=1)
  - GRL acts on 256-dim features before subject embedding is added
  - pushes features to be hard/easy-group invariant
Label head: concat(features, subject_emb) → Dropout + Linear → 50 classes
  - subject embedding is still available for gesture classification

Difference from v18: subject embedding is preserved so the model retains
per-subject adaptation ability; only the backbone features are pushed toward
group invariance, not the full representation.
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


class SubjectAwareDANN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_subjects: int):
        super().__init__()
        self.subj_emb = nn.Embedding(n_subjects, 32)

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

        # label head: features(256) + subject_emb(32) = 288
        self.label_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(288, n_classes),
        )

        # domain head: binary hard(0) vs easy(1) — operates on 256-dim features
        self.domain_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def _features(self, x):
        h = self.conv(x)               # (B, 128, T)
        h = h.transpose(1, 2)          # (B, T, 128)
        h, _ = self.lstm(h)            # (B, T, 256)
        a = torch.softmax(self.attn(h), dim=1)
        return (h * a).sum(dim=1)      # (B, 256)

    def forward(self, x, s, alpha: float = 0.0):
        feat = self._features(x)                        # (B, 256)
        emb  = self.subj_emb(s)                        # (B, 32)
        class_logits  = self.label_head(torch.cat([feat, emb], dim=1))
        domain_logits = self.domain_head(_GRL.apply(feat, alpha))
        return class_logits, domain_logits
