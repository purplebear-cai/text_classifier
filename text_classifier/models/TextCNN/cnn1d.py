from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN1D(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vocab_size: int,
                 embeddings: torch.Tensor,
                 emb_size: int,
                 fine_tune: bool,
                 n_kernels: int,
                 kernel_sizes: List[int],
                 dropout: float,
                 n_channels=1
                 ) -> None:
        """
        Implementation of 1D version of TextCNN.
        """
        super(TextCNN1D, self).__init__()

        # embedding layer
        self.embedding1 = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, 1, fine_tune)

        if n_channels == 2:
            # Multichannel: a static channel and a non-static channel
            # means that embedding2 is frozen
            self.embedding2 = nn.Embedding(vocab_size, emb_size)
            self.set_embeddings(embeddings, 1, False)
        else:
            self.embedding2 = None

        # 1D conv layer
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=n_channels,
                      out_channels=n_kernels,
                      kernel_size=size*emb_size,
                      stride=emb_size)
            for size in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def set_embeddings(self,
                       embeddings: torch.Tensor,
                       layer_id: int=1,
                       fine_tune: bool=True) -> None:
        """
        Set weights of the embedding layer in the model.
        :param embeddings: torch.Tensor, Word embeddings value.
        :param layer_id: int, Embedding layer either 1 or 2.
        :param fine_tune: bool, optional, default=True, True if allows fine-tuning of embedding layer.
        :return: None
        """
        if embeddings is None:
            # Not loading pre-trained embeddings, randomly initialize the embedding values.
            if layer_id == 1:
                self.embedding1.weight.data.uniform(-0.1, 0.1)
            else:
                self.embedding2.weight.data.uniform(-0.1, 0.1)
        else:
            # initialize embedding with pre-trained embeddings
            if layer_id == 1:
                self.embedding1.weight = nn.Parameter(embeddings, requires_grad=fine_tune)
            else:
                self.embedding2.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor,
                word_per_sentence: torch.Tensor):
        """
        NN forward.
        :param text: torch.Tensor(batch_size, word_pad_len), Input data.
        :param word_per_sentence: torch.Tensor(batch_size), Sentence lengths.
        :return: scores: torch.Tensor(batch_size, n_classes), Class scores.
        """
        batch_size = text.size(0)

        # word embedding
        embeddings = self.embedding1(text).view(batch_size, 1, -1) # (batch_size, 1, word_pad_len * emb_size)

        # multichannel
        if self.embedding2:
            embeddings2 = self.embedding2(text).view(batch_size, 1, -1)  # (batch_size, 1, word_pad_len * emb_size)
            embeddings = torch.cat((embeddings, embeddings2), dim=1)

        # conv
        conved = [self.relu(conv(embeddings)) for conv in self.convs] # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]

        # pooling
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]

        # flatten
        flattened = self.dropout(torch.cat(pooled, dim=1))  # (batch size, n_kernels * len(kernel_sizes))
        scores = self.fc(flattened)  # (batch size, n_classes)

        return scores