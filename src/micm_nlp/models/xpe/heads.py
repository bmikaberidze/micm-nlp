"""Reparameterization heads used by :class:`CrossPromptEncoder`."""

import torch

# MLP ===========================================================================


def gen_mlp_head(input_size, hidden_size, output_size, num_layers, dropout=0.1):
    """Build the MLP reparameterization head.

    ``num_layers`` counts Linear layers in total: the input projection, then
    ``num_layers - 1`` hidden blocks, then the output projection. Each hidden block
    is Linear/ReLU/Dropout; the output projection has neither activation nor
    dropout, since it produces embeddings rather than features.

    :returns: a ``torch.nn.Sequential``.
    """
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
    """Build the LSTM reparameterization head.

    A bidirectional LSTM followed by a single-layer MLP; the MLP's input is
    ``hidden_size * 2`` because the two directions are concatenated. Being
    bidirectional, each virtual token is conditioned on the whole prompt, not only
    on the tokens before it.

    :returns: an :class:`LSTMWrapper` -- the LSTM cannot be put in a ``Sequential``
        directly because it returns a tuple.
    """
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
    """Chain an LSTM into an MLP, discarding the hidden state.

    Exists because ``torch.nn.LSTM`` returns ``(output, (h, c))``, which
    ``torch.nn.Sequential`` cannot pass on.
    """

    def __init__(self, lstm: torch.nn.LSTM, mlp: torch.nn.Module):
        """:param lstm: the recurrent layer.
        :param mlp: projection applied to its output.
        """
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp

    def forward(self, x):
        """Run the LSTM, drop the hidden state, project the output sequence."""
        lstm_out, _ = self.lstm(x)
        return self.mlp(lstm_out)


# ATTN ===========================================================================


def gen_attn_head(num_heads, input_size, hidden_size, output_size, dropout):
    """Build the attention reparameterization head.

    Self-attention over the virtual tokens, then a single-layer MLP. Like the LSTM
    head this lets each virtual token see the others, but without a recurrence.

    :returns: a ``torch.nn.Sequential``.
    """
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
    """One pre-norm self-attention block: attention, residual, LayerNorm, projection.

    "Lightweight" is literal -- a single block with no feed-forward sublayer, which
    is all the prompt needs. It is the attention variant of the reparameterization
    head, selected by
    :class:`~micm_nlp.models.xpe.enums.CrossPromptEncoderReparameterizationType`.
    """

    def __init__(self, num_heads: int, embed_dim: int, output_dim: int, dropout: float = 0.1):
        """:param num_heads: attention heads; must divide ``embed_dim``.
        :param embed_dim: width of the incoming virtual-token embeddings.
        :param output_dim: width to project to.
        :param dropout: dropout on the attention output, before the residual.
        """
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
        """:param x: ``(batch_size, num_tokens, embed_dim)``.
        :returns: ``(batch_size, num_tokens, output_dim)``.
        """
        # x: (batch_size, num_tokens, hidden_size)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.layernorm(x + self.dropout(attn_output))
        x = self.out_proj(x)
        return x
