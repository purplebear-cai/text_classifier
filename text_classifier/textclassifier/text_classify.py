import os
from typing import Dict, Tuple

import torch
from nltk import TreebankWordTokenizer
from torch import nn

from text_classifier.dataloader import get_label_map
from text_classifier.utils.commons import load_checkpoint
from text_classifier.utils.opts import parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepro_sent(text: str,
                word_map: Dict[str, int],
                word_limit: int,
                ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Encode a sentence.
    :param text: str, Input raw text.
    :param word_map: Dict, Mapping from word to index.
    :param word_limit: int, Maximum number of words.
    :return:Tuple[torch.LongTensor, torch.LongTensor], (preprocessed tokenized sentence, sentence lengths)
    """
    word_tokenizer = TreebankWordTokenizer()
    sentence = word_tokenizer.tokenize(text)[:word_limit]
    word_per_sentence = len(sentence)
    word_per_sentence = torch.LongTensor([word_per_sentence]).to(device)
    encoded_sentence = list(map(lambda w: word_map.get(w, word_map['<unk>']), sentence)) \
                       + [0] * (word_limit - word_per_sentence)
    encoded_sentence = torch.LongTensor(encoded_sentence).unsqueeze(0).to(device)
    return encoded_sentence, word_per_sentence


def classify(text: str,
             model: nn.Module,
             model_name: str,
             dataset_name: str,
             word_map: Dict[str, int],
             word_limit: int) -> Tuple[str, float]:
    """
    Classify a text given the model.
    :param text: str, Raw text input.
    :param model: nn.Torch, Pre-trained model.
    :param model_name: str, Pre-trained model name.
    :param dataset_name: str, Dataset name.
    :param word_map: Dict[str, int], Mapping from word to idx.
    :param word_limit: int, Maximum number of words.
    :return: (str, float), (predicted_label, predicted_score).
    """
    _, rev_label_map = get_label_map(dataset_name)
    encoded_sent, word_per_sentence = prepro_sent(text, word_map, word_limit)
    scores = model(encoded_sent, word_per_sentence)

    scores = scores.squeeze(0)
    scores = nn.functional.softmax(scores, dim=0) # convert to probability
    score, prediction = scores.max(dim=0)
    predicted_label = rev_label_map[prediction.item()]
    return predicted_label, score


if __name__ == '__main__':
    # load config
    config = parse_opt()
    word_limit = config.word_limit

    # load checkpoint
    checkpoint_path = os.path.join(config.checkpoint_path, config.checkpoint_basename + '.pth.tar')
    model, model_name, optimizer, dataset_name, word_map, start_epoch = load_checkpoint(checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # loop test
    text = ''
    while text not in ['quit', 'exit']:
        text = input('\n text: > ')
        predicted_label, predicted_score = classify(
            text, model, model_name, dataset_name, word_map, word_limit)
        print('\npredicted_label={}\npredicted_score={}'.format(predicted_label, predicted_score))
