"""PyTorch intent classification models for CampusMate."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention pooling with padding masks."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled context and attention weights."""
        energy = torch.tanh(self.proj(outputs))
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class CampusIntentClassifier(nn.Module):
    """Hybrid CNN-BiLSTM-attention classifier, with ablation modes."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        conv_filters: int = 100,
        lstm_hidden: int = 128,
        dropout_emb: float = 0.3,
        dropout_fc: float = 0.4,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if mode not in {"hybrid", "cnn", "bilstm"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout_emb)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, conv_filters, kernel_size=k) for k in (2, 3, 4)]
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = AdditiveAttention(lstm_hidden * 2)
        cnn_dim = conv_filters * 3 if mode in {"hybrid", "cnn"} else 0
        lstm_dim = lstm_hidden * 2 if mode in {"hybrid", "bilstm"} else 0
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(cnn_dim + lstm_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Compute logits and optionally return attention weights."""
        mask = x.ne(0)
        emb = self.embed_dropout(self.embedding(x))
        features: list[torch.Tensor] = []
        attn_weights = torch.zeros(x.size(0), x.size(1), device=x.device)
        if self.mode in {"hybrid", "cnn"}:
            conv_in = emb.transpose(1, 2)
            pooled = [F.max_pool1d(F.relu(conv(conv_in)), kernel_size=conv(conv_in).shape[-1]).squeeze(-1) for conv in self.convs]
            features.append(torch.cat(pooled, dim=1))
        if self.mode in {"hybrid", "bilstm"}:
            outputs, _ = self.lstm(emb)
            context, attn_weights = self.attention(outputs, mask)
            features.append(context)
        combined = torch.cat(features, dim=1)
        hidden = F.relu(self.fc1(self.dropout(combined)))
        logits = self.fc2(hidden)
        if return_attention:
            return logits, attn_weights
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
