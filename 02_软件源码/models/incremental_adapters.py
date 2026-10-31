"""Small residual adapters for a frozen incremental detector."""

from torch import nn


class ResidualAdapter(nn.Module):
    """Start as an exact identity; learn a low-dimensional feature update."""

    def __init__(self, hidden_dim, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        if not 0 < bottleneck_dim <= hidden_dim:
            raise ValueError("adapter_dim must be in (0, hidden_dim]")
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        update = self.up(self.dropout(self.activation(self.down(self.norm(features)))))
        return features + update


class DecoupledHeadAdapters(nn.Module):
    """Share each task's adapter across decoder levels to limit parameters."""

    def __init__(self, hidden_dim, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        self.classification = ResidualAdapter(hidden_dim, bottleneck_dim, dropout)
        self.localization = ResidualAdapter(hidden_dim, bottleneck_dim, dropout)

    def forward(self, features):
        return self.classification(features), self.localization(features)


def is_adapter_parameter(name):
    return name.startswith("head_adapters.") or (
        name.startswith("transformer.decoder.layers.") and ".adapter." in name
    )
