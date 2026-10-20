import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # 
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
