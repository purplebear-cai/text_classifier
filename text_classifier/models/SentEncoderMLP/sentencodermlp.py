import torch
from sentence_transformers import SentenceTransformer
from torch import nn


class SentEncoderMLP(nn.Module):
    """
    Multi-layer perceptron model.
    :param n_classes: int, Number of labels
    :param encoder_name: str, The reference name for pretrained transformer model.
    :param embed_size: int, Pretrained embedding size.
    :param hidden_size: int, Output dimension of Dense layer.
    """
    def __init__(self,
                 n_classes: int,
                 encoder_name: str,
                 embed_size: int,
                 hidden_size: int) -> None:
        super(SentEncoderMLP, self).__init__()

        # embedding layer
        self.embeddings = SentenceTransformer(encoder_name)

        # mlp layer
        self.hidden_fc = nn.Linear(embed_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, n_classes)

    def forward(self, text):
        """
        NN forward.
        :param text: torch.Tensor(batch_size, word_pad_length), Input data.
        :return: scores: torch.Tensor(batch_size, n_classes), Class scores.
        """
        # [batch_size, 1, -1]
        embeddings = self.embeddings.encode(text)
        embeddings = torch.from_numpy(embeddings)

        # [batch_size, hidden_size]
        hidden = self.hidden_fc(embeddings)

        # [batch_size, n_classes]
        scores = self.out_fc(hidden)

        return scores