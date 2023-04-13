import os
import torch
from pandas import DataFrame

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from text_classifier.dataloader import load_data, get_label_map
from text_classifier.utils.commons import load_checkpoint, AverageMeter
from text_classifier.utils.confusion_matrix_plotter import pretty_plot_confusion_matrix
from text_classifier.utils.opts import parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model: nn.Module,
         model_name: str,
         test_loader: DataLoader,
         dataset_name: str,
         plot_cm=False) -> None:
    """
    Corpus test on given test data.
    :param model: nn.Module, Specific NN model.
    :param model_name: str, Specific NN model name.
    :param test_loader: DataLoader, Data loader.
    :param dataset_name: str, The name of dataset.
    :param plot_cm: bool, True if plot confusion matrix, otherwise False
    :return: None
    """
    # metrics tracking
    accs = AverageMeter()
    label_map, _ = get_label_map(dataset_name)
    n_classes = len(label_map)
    confusion_matrix = torch.zeros(n_classes, n_classes)

    # evaluate in batches
    with torch.no_grad():
        # Context-manager that disabled gradient calculation
        for i, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            sentences_text, sentences, words_per_sentence, labels = batch

            # move data to specific device
            # squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
            sentences = sentences.to(device)
            words_per_sentence = words_per_sentence.squeeze(1).to(device)
            labels = labels.squeeze(1).to(device)

            # compute scores, return a Tensor[batch_size, n_classes]
            if model_name == 'sentencoder_mlp':
                scores = model(sentences_text)
            else:
                scores = model(sentences, words_per_sentence)

            # compute accuracy
            _, predictions = scores.max(dim=1)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # add accuracy to tracking metrics
            accs.update(accuracy, labels.size(0))

            # update confusion matrix
            for true_label, predicted_label in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[true_label.long(), predicted_label.long()] += 1

        # print final test accuracy
        print('\nTest Accuracy - %.1f' % (accs.avg * 100))

        # print confusion matrix
        if plot_cm:
            print('\nConfusion matrix:')
            _, rev_label_map = get_label_map(dataset_name)
            cm_df = DataFrame(confusion_matrix.numpy(),
                              index=[rev_label_map.get(i) for i in range(0, n_classes)],
                              columns=[rev_label_map.get(i) for i in range(0, n_classes)])
            pretty_plot_confusion_matrix(cm_df, cmap='PuRd')


if __name__ == '__main__':
    # load config
    config = parse_opt()

    # load model
    checkpoint_path = os.path.join(config.checkpoint_path, config.checkpoint_basename + '.pth.tar')
    model, model_name, _, dataset_name, _, _ = load_checkpoint(checkpoint_path, device)
    model = model.to(device)
    model.eval() # Sets the module in evaluation mode.

    # load test data
    test_loader = load_data(config, "test")

    # start testing
    test(model, model_name, test_loader, dataset_name)
