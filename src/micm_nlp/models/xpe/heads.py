"""Reparameterization heads used by :class:`CrossPromptEncoder`."""

import torch

# MLP ===========================================================================


def gen_mlp_head(input_size, hidden_size, output_size, num_layers, dropout=0.1):
    hidden_layers = []
    for _ in range(num_layers - 1):
        hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        hidden_layers.append(torch.nn.ReLU())
        hidden_layers.append(torch.nn.Dropout(dropout))
    layers = [
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        *hidden_layers,
        torch.nn.Linear(hidden_size, output_size),
    ]
    mlp_head = torch.nn.Sequential(*layers)
    return mlp_head


# LSTM ===========================================================================


def gen_lstm_head(input_size, hidden_size, output_size, num_layers, dropout):
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        batch_first=True,
    )
    mlp = gen_mlp_head(
        input_size=hidden_size * 2,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=1,
        dropout=dropout,
    )
    return LSTMWrapper(lstm, mlp)


class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm: torch.nn.LSTM, mlp: torch.nn.Module):
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.mlp(lstm_out)


# ATTN ===========================================================================


def gen_attn_head(num_heads, input_size, hidden_size, output_size, dropout):
    attn = LightweightSelfAttentionHead(
        num_heads=num_heads,
        embed_dim=input_size,
        output_dim=input_size,
        dropout=dropout,
    )
    mlp = gen_mlp_head(
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
    )
    return torch.nn.Sequential(attn, mlp)


class LightweightSelfAttentionHead(torch.nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm = torch.nn.LayerNorm(embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, num_tokens, hidden_size)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.layernorm(x + self.dropout(attn_output))
        x = self.out_proj(x)
        return x
