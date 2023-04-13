
import time
import torch
import mlflow
import mlflow.pytorch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

from text_classifier.utils.commons import AverageMeter, clip_gradient, adjust_learning_rate, save_checkpoint
from text_classifier.utils.tensorboard import TensorboardLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self,
                 num_epochs: int,
                 start_epoch: int,
                 train_loader: DataLoader,
                 model: nn.Module,
                 model_name: str,
                 loss_function: nn.Module,
                 optimizer,
                 lr_decay: float,
                 dataset_name: str,
                 word_map: Dict[str, int],
                 grad_clip=Optional[None],
                 print_freq: int = 100,
                 checkpoint_path: Optional[str] = None,
                 checkpoint_basename: str = 'checkpoint',
                 tensorboard: bool = False,
                 log_dir: Optional[str] = None
                 ) -> None:
        """
        Training pipeline
        :param num_epochs: int, Number for training epochs.
        :param start_epoch: int, Start training from the __th epoch.
        :param train_loader: DataLoader, Training data loader.
        :param model: nn.Module, Classifier model.
        :param model_name: str, Model name.
        :param loss_function: nn.Module, Loss function.
        :param optimizer: optim.Optimizer, Optimizer (e.g. Adam).
        :param lr_decay: float, a factor in interval (0,1) to multiply with the learning rate.
        :param dataset_name: str, Name of the dataset.
        :param word_map: Dict, Mapping from word to idx.
        :param grad_clip: float, optional, Gradient threshold in clip gradients.
        :param print_freq: int, optional, Print training status every <print_freq> batches.
        :param checkpoint_path: str, Path to the folder to save checkpoint.
        :param checkpoint_basename: str, Checkpoint name.
        :param tensorboard: bool, optional, Enable tensorboard or not.
        :param log_dir: str, optional, Path to the folder to save logs.
        """
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.train_loader = train_loader

        self.model = model
        self.model_name = model_name
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_decay = lr_decay

        self.dataset_name = dataset_name
        self.word_map = word_map
        self.print_freq = print_freq
        self.grad_clip = grad_clip

        self.checkpoint_path = checkpoint_path
        self.checkpoint_basename = checkpoint_basename

        # setup visualization writer instance
        self.writer = TensorboardLogger(log_dir, tensorboard)
        self.len_epoch = len(self.train_loader)

    def train(self, epoch: int) -> None:
        """
        Training pipeline for one epoch.
        :param epoch: int, Current number of epoch.
        :return: None
        """
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(tag="loss", writer=self.writer)
        accs = AverageMeter(tag="acc", writer=self.writer)

        start = time.time()

        for i, batch in enumerate(self.train_loader):
            data_time.update(time.time() - start)
            if self.model_name in ["han"]:
                raise ValueError("Not Implemented Model Found for {}".format(self.model_name))
            else:
                sentences_text, sentences, word_per_sentences, labels = batch
                sentences = sentences.to(device)
                word_per_sentences = word_per_sentences.squeeze(1).to(device)
                labels = labels.squeeze(1).to(device)
                if self.model_name == "sentencoder_mlp":
                    # map raw sentence to embeddings calling pre-trained encoder
                    scores = self.model(sentences_text)
                else:
                    scores = self.model(sentences, word_per_sentences)

            # calculate loss
            loss = self.loss_function(scores, labels)

            # backward update
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, self.grad_clip)

            # update weights
            self.optimizer.step()

            # calculate accuracy
            _, predictions = scores.max(dim=1)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # set step for tensorboard
            step = (epoch - 1) * self.len_epoch + i
            self.writer.set_step(step=step, mode="train")

            # keep track of the metrics
            batch_time.update(time.time() - start)
            losses.update(loss.item(), labels.size(0))
            accs.update(accuracy, labels.size(0))

            start = time.time()

            # print training status
            if i % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accs
                    )
                )

        mlflow.log_metric(key="loss", value=losses.avg, step=epoch)
        mlflow.log_metric(key="accuracy", value=accs.avg, step=epoch)

    def run_train(self):
        """
        Training pipeline.
        """
        start = time.time()

        # epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # train for the current epoch
            self.train(epoch=epoch)

            # time per epoch
            epoch_time = time.time() - start
            print('Epoch: [{0}] finished, time consumed: {epoch_time:.3f}'.format(epoch, epoch_time=epoch_time))

            # decay learning rate every epoch
            adjust_learning_rate(self.optimizer, self.lr_decay)

            # save chekcpoints
            if self.checkpoint_path is not None:
                save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    model_name=self.model_name,
                    optimizer=self.optimizer,
                    dataset_name=self.dataset_name,
                    word_map=self.word_map,
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_basename=self.checkpoint_basename
                )
                mlflow.log_artifact(self.checkpoint_path)