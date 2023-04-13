import torch
from torch import nn


class TextMLP(nn.Module):
    """
    Multi-layer perceptron model.
    :param n_classes:
    :param vocab_size:
    :param embeddings:
    :param emb_size:
    :param fine_tune:
    :param layers: int, Number of Dense layers.
    :param units: int, Output dimension of Dense layer.
    :param dropout:
    """
    def __init__(self,
                 n_classes: int,
                 vocab_size: int,
                 embeddings: torch.Tensor,
                 emb_size: int,
                 fine_tune: bool,
                 hidden_size: int) -> None:

        super(TextMLP, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # mlp layer
        # TODO: smarter way
        self.hidden_fc = nn.Linear(emb_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, n_classes)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool):
        """
        Set weights of the embedding layer in the model.
        :param embeddings: torch.Tensor, Word embeddings value.
        :param fine_tune: bool, optional, default=True, True if allows fine-tuning of embedding layer.
        :return: None
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor):
        """
        NN forward.
        :param text: torch.Tensor(batch_size, word_pad_length), Input data.
        :return: scores: torch.Tensor(batch_size, n_classes), Class scores.
        """
        # [batch_size, 1, -1]
        embeddings = self.embeddings(text)

        # [batch_size, embed_size]
        avg_embeddings = embeddings.mean(dim=1).squeeze(1)

        # [batch_size, hidden_size]
        hidden = self.hidden_fc(avg_embeddings)

        # [batch_size, n_classes]
        scores = self.out_fc(hidden)

        return scores