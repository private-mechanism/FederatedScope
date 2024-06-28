import logging
from typing import Type
import random

import torch
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


def wrap_LabelFlippingTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    wrap the label flipping trainer

    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer
    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer
    '''

    base_trainer.replace_hook_in_train(
        new_hook=_hook_on_batch_forward_with_flipped_labels,
        target_trigger='on_batch_forward',
        target_hook_name='_hook_on_batch_forward')

    return base_trainer


# def hook_on_batch_backward_generate_gaussian_noise_gradient(ctx):
#     ctx.optimizer.zero_grad()
#     ctx.loss_task.backward()

#     grad_values = list()
#     for name, param in ctx.model.named_parameters():
#         if 'bn' not in name:
#             grad_values.append(param.grad.detach().cpu().view(-1))

#     grad_values = torch.cat(grad_values)
#     # mean_for_gaussian_noise = torch.mean(grad_values) + 0.1
#     mean_for_gaussian_noise = 0
#     std_for_gaussian_noise = torch.std(grad_values)

#     for name, param in ctx.model.named_parameters():
#         if 'bn' not in name:
#             generated_grad = torch.normal(mean=mean_for_gaussian_noise,
#                                           std=std_for_gaussian_noise,
#                                           size=param.grad.shape)
#             param.grad = generated_grad.to(param.grad.device)

#     ctx.optimizer.step()

def _hook_on_batch_forward_with_flipped_labels(ctx):
    """
    Note:
        The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.y_true``                      Move to `ctx.device`
        ``ctx.y_prob``                      Forward propagation get y_prob
        ``ctx.loss_batch``                  Calculate the loss
        ``ctx.batch_size``                  Get the batch_size
        ==================================  ===========================
    """
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]
    fake_label = ctx.cfg.model.out_channels - 1 - label
    # fake_label = torch.tensor(torch.randint_like(label, 0, int(ctx.cfg.model.out_channels-1)))
    pred = ctx.model(x)
    if len(fake_label.size()) == 0:
        fake_label = fake_label.unsqueeze(0)
    ctx.y_true = CtxVar(fake_label, LIFECYCLE.BATCH)
    ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
    ctx.loss_batch = CtxVar(ctx.criterion(pred, fake_label), LIFECYCLE.BATCH)
    ctx.batch_size = CtxVar(len(fake_label), LIFECYCLE.BATCH)