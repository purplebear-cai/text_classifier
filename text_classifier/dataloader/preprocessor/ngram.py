"""Module to vectorize data.

Converts the given training and validation texts into numerical tensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import logging
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from collections import Counter
from text_classifier.utils.text import get_clean_text
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

word_tokenizer = TreebankWordTokenizer()

logger = logging.getLogger("ngram_feature_extractor")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
MAX_FEATURES = 10000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500


def run_prepro(csv_folder: str,
               output_folder: str,
               word_limit: int,
               min_word_count: int=5) -> None:
    # --------------------- Create keyword arguments to pass to the 'tf-idf' vectorizer. ---------------------
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
        'max_features': MAX_FEATURES
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # --------------------- training data ---------------------
    print('\nTraining data: reading and preprocessing...\n')
    train_sents_text, train_sents, train_labels, word_counter = read_csv(csv_folder, 'train', word_limit)
    words_per_sentence = list(map(lambda s: len(s), train_sents))
    x_train = vectorizer.fit_transform(train_sents_text).todense()

    # save word map
    filename = os.path.join(output_folder, 'word_map.json')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as j:
        json.dump({k:int(v) for k, v in vectorizer.vocabulary_.items()}, j)
    print('Training data: word map saved to %s.\n' % os.path.abspath(output_folder))

    print('Training data: saving...\n')
    # because of the large data, saving as a JSON can be very slow
    torch.save({
        'sents_text': train_sents_text, #
        'sents': x_train.tolist(), # x_train.tolist()
        'labels': train_labels,
        'words_per_sentence': words_per_sentence
    }, os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Training data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    # --------------------- training data ---------------------
    test_sents_text, test_sents, test_labels, _ = read_csv(csv_folder, 'test', word_limit)

    missing_classes = [i for i in test_labels if i not in train_labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}.'.format(missing_classes=missing_classes))

    words_per_sentence = list(map(lambda s: len(s), test_sents))
    x_test = vectorizer.transform(test_sents_text).todense()

    print('Test data: saving...\n')
    torch.save({
        'sents_text': test_sents_text, #
        'sents': x_test.tolist(), # x_test.tolist()
        'labels': test_labels,
        'words_per_sentence': words_per_sentence
    }, os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Test data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    print('All done!\n')


def read_csv(csv_folder: str, split: str, word_limit: int) \
        -> Tuple[list, list, Counter]:
    """
    Read CSVs containing raw training data, clean sentences and labels, and do a word-count.
    :param csv_folder: str, Folder containing the dataset in CSV format files.
    :param split: str, either 'train' or 'test'.
    :param word_limit: int, Truncate the long sentence to the maximum limit number.
    :return:
        sents: list, List of sentences, each sentence is defined as [word1, word2, ..., wordn]
        labels: list, List of label for each sentence.
        word_counter: Counter, Word frequency counts.
    """
    assert split in {'train', 'test'}

    sents_text = []
    sents = []
    labels = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header=1)

    # # TODO: for debugging
    # data = data.head(200)

    for i in tqdm(range(data.shape[0])):
        row = list(data.loc[i, :])

        s = ''

        for text in row[1:]:
            text = get_clean_text(text)
            s = s + text

        sents_text.append(s)
        words = word_tokenizer.tokenize(s)
        # if sentence is empty (due to removing punctuation, digits, etc.)
        if len(words) == 0:
            continue
        word_counter.update(words)

        labels.append(int(row[0]) - 1) # since labels are 1-indexed in the CSV
        sents.append(words)

    return sents_text, sents, labels, word_counter