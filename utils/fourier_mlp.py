import math
from typing import Tuple

import torch
from torch import nn


class FourierFeatureMLP(nn.Module):
    """MLP with Gaussian Fourier feature encoding for coordinate inputs."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        mapping_size: int = 256,
        scale: float = 10.0,
        include_input: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mapping_size = mapping_size
        self.include_input = include_input

        # Fixed random Fourier projection matrix used to map low-d coords to high-d features.
        B = torch.randn(dim_in, mapping_size) * scale
        self.B: torch.Tensor
        self.register_buffer("B", B)

        encoded_dim = (2 * mapping_size) + (dim_in if include_input else 0)
        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(encoded_dim, dim_out))
        else:
            layers.append(nn.Linear(encoded_dim, dim_hidden))
            layers.append(self._build_activation(activation))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(dim_hidden, dim_hidden))
                layers.append(self._build_activation(activation))
            layers.append(nn.Linear(dim_hidden, dim_out))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _build_activation(self, activation: str) -> nn.Module:
        activation_l = activation.lower()
        if activation_l == "relu":
            return nn.ReLU()
        if activation_l == "gelu":
            return nn.GELU()
        if activation_l == "silu":
            return nn.SiLU()
        if activation_l == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Unsupported Fourier MLP activation '{activation}'. "
            "Use one of: relu, gelu, silu, tanh."
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim_in], projected: [N, mapping_size]
        projected = 2.0 * math.pi * torch.matmul(x, self.B)
        encoded = [torch.sin(projected), torch.cos(projected)]
        if self.include_input:
            encoded.append(x)
        return torch.cat(encoded, dim=-1)

    def _flatten_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if x.shape[-1] != self.dim_in:
            raise ValueError(
                f"Expected last input dim {self.dim_in}, but got {x.shape[-1]}"
            )
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        return x_flat, original_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, original_shape = self._flatten_input(x)
        encoded = self._encode(x_flat)
        out = self.mlp(encoded)
        return out.reshape(*original_shape, self.dim_out)
