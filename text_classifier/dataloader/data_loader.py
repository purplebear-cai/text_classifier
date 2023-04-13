import os
import json
import torch
from typing import Union, Tuple, Dict
from torch.utils.data import DataLoader, Dataset

from text_classifier.dataloader.info import get_label_map

from text_classifier.utils.embeddings import load_embeddings
from text_classifier.utils.opts import Config


class DocDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches (for document classification).
    """
    def __init__(self, data_folder: str, split: str) -> None:
        """
        Initializer.
        :param data_folder: str, Path to folder where data files are stored
        :param split: str, either 'train' or 'test'
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


class NgramDataset(Dataset):
    def __init__(self, data_folder: str, split: str) -> None:
        """
        A Pytorch Dataset class to be used in a PyTorch DataLoader to create batches, for sentence classification.
        :param data_folder: str, Path to data files.
        :param split: str, either 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return self.data['sents_text'][i], \
               torch.FloatTensor(self.data['sents'][i]), \
               torch.LongTensor([self.data['words_per_sentence'][i]]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


class SentDataset(Dataset):
    def __init__(self, data_folder: str, split: str) -> None:
        """
        A Pytorch Dataset class to be used in a PyTorch DataLoader to create batches, for sentence classification.
        :param data_folder: str, Path to data files.
        :param split: str, either 'TRAIN' or 'TEST'
        :param model_name: str, if model_name == ngram_mlp, sents is returned as FloatTensor,
        otherwise sents is returned as LongTensor
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:

        return self.data['sents_text'][i], \
               torch.LongTensor(self.data['sents'][i]), \
               torch.LongTensor([self.data['words_per_sentence'][i]]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


def load_data(
    config: Config, split: str, build_vocab: bool = True) \
        -> Union[DataLoader, Tuple[DataLoader, torch.Tensor, int, Dict[str, int], int, int]]:
    """
    Load data from files output by ``prepocess.py``.
    :param config: Config, configuration settings.
    :param split: str, either 'train' or 'test'
    :param build_vocab: bool, True if build vocabulary on the fly, only makes sense when split=='train'.
    :return:
    if split == 'test':
        returns: test_loader: DataLoader, DataLoader for test data.
    if split == 'train':
        if build_loader == False:
            returns: train_loader: DataLoader, DataLoader for train data.
        if build_loader == True:
            returns: train_loader: DataLoader, DataLoader for train data.
                     embeddings: torch.Tensor, pre-trained word embeddings (None if config.emb_pretrain==False).
                     emb_size: int, embedding size (config.emb_size if config.emb_pretrain==False).
                     word_map: Dict[str, int], mapping form word to idx.
                     n_classes : int, Number of classes
                    vocab_size : int, Size of vocabulary
    """
    split = split.lower()
    assert split in {'train', 'test'}

    # test
    if split == 'test':
        if config.model_name == "han":
            dataset = DocDataset(config.output_path, 'test')
        elif config.model_name == "ngram_mlp":
            dataset = NgramDataset(config.output_path, 'test')
        else:
            dataset = SentDataset(config.output_path, 'test')
        test_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True
        )
        return test_loader

    # train
    else:
        # dataloaders
        if config.model_name == "han":
            dataset = DocDataset(config.output_path, 'train')
        elif config.model_name == "ngram_mlp":
            dataset = NgramDataset(config.output_path, 'train')
        else:
            dataset = SentDataset(config.output_path, 'train')

        train_loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle = True,
            num_workers = config.workers,
            pin_memory = True
        )

        if build_vocab == False:
            # return train_loader
            # number of classes
            label_map, _ = get_label_map(config.dataset)
            n_classes = len(label_map)

            return train_loader, None, None, None, n_classes, None

        else:
            # load word2ix map
            with open(os.path.join(config.output_path, 'word_map.json'), 'r') as j:
                word_map = json.load(j)
            # size of vocabulary
            vocab_size = len(word_map)

            # number of classes
            label_map, _ = get_label_map(config.dataset)
            n_classes = len(label_map)

            if config.model_name == "ngram_mlp":
                emb_size = vocab_size
                return train_loader, None, emb_size, None, n_classes, None

            # word embeddings
            if config.encoder_name is not None:
                embeddings = None
                emb_size = config.emb_size

            elif config.emb_pretrain == True:
                # load Glove as pre-trained word embeddings for words in the word map
                emb_path = os.path.join(config.emb_folder, config.emb_filename)
                embeddings, emb_size = load_embeddings(
                    emb_file = os.path.join(config.emb_folder, config.emb_filename),
                    word_map = word_map,
                    output_folder = config.output_path
                )
            # or initialize embedding weights randomly
            else:
                embeddings = None
                emb_size = config.emb_size

            return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size
