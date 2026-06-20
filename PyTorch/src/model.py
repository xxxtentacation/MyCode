from __future__ import annotations

import torch
from torch import nn


class TinyTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = masked_embeddings.sum(dim=1) / lengths
        logits = self.classifier(self.dropout(pooled))
        return logits


class ImprovedTextClassifier(nn.Module):
    """Text classifier with positional encoding, multi-kernel CNNs, and self-attention pooling."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 64,
        num_filters: int = 48,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.5,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout * 0.5)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        self.conv_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)
        self.norm = nn.LayerNorm(total_filters)
        self.attn = nn.Linear(total_filters, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 3, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape

        emb = self.embedding(input_ids)
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        emb = emb + self.pos_embedding(positions)
        emb = self.emb_dropout(emb)
        emb = emb * attention_mask.unsqueeze(-1)

        # Multi-kernel 1D convolutions: (B, E, L) -> (B, F, L) each
        emb_t = emb.transpose(1, 2)
        conv_outputs = [torch.relu(bn(conv(emb_t))) for conv, bn in zip(self.convs, self.conv_norms)]
        cat = torch.cat(conv_outputs, dim=1).transpose(1, 2)  # (B, L, total_F)
        cat = self.norm(cat)

        # Self-attention pooling
        scores = self.attn(cat)
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (cat * weights).sum(dim=1)  # (B, total_F)

        return self.classifier(self.dropout(pooled))


def create_model(model_type: str, vocab_size: int, num_labels: int, **kwargs) -> nn.Module:
    if model_type == "tiny":
        return TinyTextClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            embed_dim=kwargs.get("embed_dim", 64),
            dropout=kwargs.get("dropout", 0.1),
        )
    if model_type == "improved":
        return ImprovedTextClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            embed_dim=kwargs.get("embed_dim", 64),
            num_filters=kwargs.get("num_filters", 48),
            kernel_sizes=kwargs.get("kernel_sizes", (3, 5, 7)),
            dropout=kwargs.get("dropout", 0.5),
            max_seq_len=kwargs.get("max_seq_len", 128),
        )
    raise ValueError(f"Unknown model_type: {model_type}")