import os
import torch
from torch import nn, optim
from typing import Tuple, Dict


def load_checkpoint(checkpoint_path: str, device: torch.device) \
        -> Tuple[nn.Module, str, optim.Optimizer, str, Dict[str, int], int]:
    """
    Load a checkpoint, so that we can continue to train on it

    :param checkpoint_path: str, Path to the checkpoint to be loaded.
    :param device: torch.device, Remap the model to which device.
    :returns:
        model: nn.Module, Model.
        model_name: str, Name of the model.
        optimizer: optim.Optimizer, Optimizer to update the model's weights.
        dataset_name: str, Name of the dataset.
        word_map: Dict[str, int], Word2ix map.
        start_epoch : int, We should start training the model from __th epoch.
    """
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1

    return model, model_name, optimizer, dataset_name, word_map, start_epoch


def adjust_learning_rate(optimizer: optim.Optimizer, scale_factor: float) -> None:
    """
    Shrink learning rate by a specified factor.
    :param optimizer: optim.Optimizer, Optimizer with the gradients to be clipped.
    :param scale_factor: float, Factor in interval (0,1) to multiply with learning rate.
    :return:
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(epoch: int,
                    model: nn.Module,
                    model_name: str,
                    optimizer: optim.Optimizer,
                    dataset_name: str,
                    word_map: Dict[str, int],
                    checkpoint_path: str,
                    checkpoint_basename: str = 'checkpoint'
                    ) -> None:
    """
    Save a model checkpoint.
    :param epoch: int, Epoch number the current checkpoint have been trained for.
    :param model: nn.Module, Model.
    :param model_name: str, Name of the model.
    :param optimizer: optim.Optimizer, Optimizer to update the model's weights.
    :param dataset_name: str, Dataset name.
    :param word_map: Dict[str, int], Mapping from word to index.
    :param checkpoint_path: str, Path to save the checkpoint.
    :param checkpoint_basename: str, Basename of the checkpoint.
    :return: None
    """
    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'dataset_name': dataset_name,
        'word_map': word_map
    }
    save_path = os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def clip_gradient(optimizer: optim.Optimizer, grad_clip: float) -> None:
    """
    Compute clip gradients during backpropagation to avoid explosion of gradients.
    :param optimizer: optim.Optimizer, Optimizer with the gradients to be clipped.
    :param grad_clip: float, Gradient clip value.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter:
    """
    Keep track of the most recent, average, sum and count of a metric.
    """
    def __init__(self, tag=None, writer=None):
        self.writer = writer
        self.tag = tag
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.writer is not None:
            self.writer.add_scalar(self.tag, val)