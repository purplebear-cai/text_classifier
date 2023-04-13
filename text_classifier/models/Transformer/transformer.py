import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
            self,
            n_classes: int,
            vocab_size: int,
            embeddings: torch.Tensor,
            d_model: torch.Tensor,
            word_pad_len: int,
            fine_tune: bool,
            hidden_size: int,
            n_heads: int,
            n_encoders: int,
            dropout: float = 0.5
    ) -> None:
        raise NotImplementedError('Not Implemented.')