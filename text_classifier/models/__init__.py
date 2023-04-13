import torch

from text_classifier.utils.opts import Config
from text_classifier.models.MLP import TextMLP
from text_classifier.models.NgramMLP.ngram_mlp import NgramMLP
from text_classifier.models.SentEncoderMLP import SentEncoderMLP
from text_classifier.models.TextCNN import TextCNN1D, TextCNN2D
from text_classifier.models.Transformer import Transformer


def make(config: Config,
         n_classes: int,
         vocab_size: int,
         embeddings: torch.Tensor,
         emb_size: int,
    ) -> torch.nn.Module:
    """
    Make a model.
    :param config: Config, Configuration settings
    :param n_classes: int, Number of classes
    :param vocab_size: int, Size of vocabulary
    :param embeddings: torch.Tensor, Word embedding weights
    :param emb_size: int, Size of word embeddings
    """
    if config.model_name == 'textcnn':
        if config.conv_layer == '2D':
            model = TextCNN2D(n_classes=n_classes,
                              vocab_size=vocab_size,
                              embeddings=embeddings,
                              emb_size=emb_size,
                              fine_tune=config.fine_tune_word_embeddings,
                              n_kernels=config.n_kernels,
                              kernel_sizes=config.kernel_sizes,
                              n_channels=config.n_channels,
                              dropout=config.dropout)
        elif config.conv_layer == '1D':
            model = TextCNN1D(n_classes=n_classes,
                              vocab_size=vocab_size,
                              embeddings=embeddings,
                              emb_size=emb_size,
                              fine_tune=config.fine_tune_word_embeddings,
                              n_kernels=config.n_kernels,
                              kernel_sizes=config.kernel_sizes,
                              n_channels=config.n_channels,
                              dropout=config.dropout)
        else:
            raise Exception("Convolution layer not supported: ", config.conv_layer)
    elif config.model_name == "ngram_mlp":
        model = NgramMLP(n_classes=n_classes,
                         input_size=emb_size,
                         hidden_size=config.hidden_size)
    elif config.model_name == 'mlp':
        model = TextMLP(n_classes=n_classes,
                        vocab_size=vocab_size,
                        embeddings=embeddings,
                        emb_size=emb_size,
                        fine_tune=config.fine_tune_word_embeddings,
                        hidden_size=config.hidden_size)
    elif config.model_name == 'transformer':
        model = Transformer(n_classes=n_classes,
                            vocab_size=vocab_size,
                            embeddings=embeddings,
                            d_model=emb_size,
                            word_pad_len=config.word_limit,
                            fine_tune=config.fine_tune_word_embeddings,
                            hidden_size=config.hidden_size,
                            n_heads=config.n_heads,
                            n_encoders=config.n_encoders,
                            dropout=config.dropout)
    elif config.model_name == 'sentencoder_mlp':
        model = SentEncoderMLP(n_classes=n_classes,
                               encoder_name=config.encoder_name,
                               embed_size=config.emb_size,
                               hidden_size=config.hidden_size)
    else:
        raise Exception("Model not supported: ", config.model_name)

    return model
