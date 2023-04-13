import torch
from torch import nn


class NgramMLP(nn.Module):
    """
    Multi-layer perceptron model.
    :param n_classes:
    :param layers: int, Number of Dense layers.
    :param units: int, Output dimension of Dense layer.
    :param dropout:
    """
    def __init__(self,
                 n_classes: int,
                 input_size: int,
                 hidden_size: int) -> None:

        super(NgramMLP, self).__init__()

        # mlp layer
        # TODO: smarter way

        # op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
        # model = models.Sequential()
        # model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
        #
        # for _ in range(layers - 1):
        #     model.add(Dense(units=units, activation='relu'))
        #     model.add(Dropout(rate=dropout_rate))
        #
        # model.add(Dense(units=op_units, activation=op_activation))


        layers=2
        # self.dropout = nn.Dropout(p=0.2)
        self.fn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, n_classes))

        # self.hidden_fc = nn.Linear(input_size, hidden_size)
        # self.out_fc = nn.Linear(hidden_size, n_classes)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor):
        # hidden = self.hidden_fc(text)
        #
        # # [batch_size, n_classes]
        # scores = self.out_fc(hidden)

        scores = self.fn(text)
        return scores
