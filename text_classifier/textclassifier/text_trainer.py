import torch
import mlflow
from torch import optim, nn

from text_classifier import models
from text_classifier.dataloader.data_loader import load_data
from text_classifier.trainer.trainer import Trainer
from text_classifier.utils.commons import load_checkpoint
from text_classifier.utils.opts import parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_trainer(config):
    """
    Set up a trainer.
    :param: obj, configurations.
    :returns: a trainer object.
    """
    if config.checkpoint is not None:
        # load existing model
        train_loader = load_data(config, 'train', False)
        model, optimizer, word_map, start_epoch = load_checkpoint(config.checkpoint, device)
        print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
    else:
        # initialize a new model
        start_epoch = 0
        train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(
            config, 'train', build_vocab=True)
        # mlflow.log_params(config.__dict__)
        mlflow.log_param('n_classes', n_classes)
        mlflow.log_param('vocab_size', vocab_size)
        model = models.make(config=config,
                            n_classes=n_classes,
                            vocab_size=vocab_size,
                            embeddings=embeddings,
                            emb_size=emb_size)
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                               lr=config.lr)
    # loss function
    loss_function = nn.CrossEntropyLoss() # TODO: qingqing, differentiate binary and multi-class classification

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    # set up trainer
    trainer = Trainer(num_epochs=config.num_epochs,
                      start_epoch=start_epoch,
                      train_loader=train_loader,
                      model=model,
                      model_name=config.model_name,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      lr_decay=config.lr_decay,
                      dataset_name=config.dataset,
                      word_map=word_map,
                      grad_clip=config.grad_clip,
                      print_freq=config.print_freq,
                      checkpoint_path=config.checkpoint_path,
                      checkpoint_basename=config.checkpoint_basename,
                      tensorboard=config.tensorboard,
                      log_dir=config.log_dir)
    return trainer


if __name__ == '__main__':
    config = parse_opt()
    trainer = set_trainer(config)
    trainer.run_train()