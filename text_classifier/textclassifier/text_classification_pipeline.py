import os
import torch
import logging
from dataloader.data_loader import load_data
from textclassifier.text_tester import test
from textclassifier.text_trainer import set_trainer
from utils.commons import load_checkpoint
from utils.opts import parse_opt
from dataloader import run_sent_prepro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format="%(filename)s: %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False


def main():
    logger.info('Parsing configuration file ...')
    config = parse_opt()

    logger.info('Preprocessing data ...')
    run_sent_prepro(
        csv_folder=config.dataset_path,
        output_folder=config.output_path,
        word_limit=config.word_limit,
        min_word_count=config.min_word_count
    )

    logger.info('Setting up training environment and starting training ...')
    trainer = set_trainer(config)
    trainer.run_train()

    logger.info('Loading trained model ...')
    checkpoint_path = os.path.join(config.checkpoint_path, config.checkpoint_basename + '.pth.tar')
    model, model_name, optimizer, dataset_name, word_map, start_epoch = load_checkpoint(checkpoint_path, device)
    model = model.to(device)
    model.eval()  # Sets the module in evaluation mode.

    logger.info('Evaluating trained model ...')
    test_loader = load_data(config, "test")
    test(model, model_name, test_loader, dataset_name)


if __name__ == '__main__':
    main()