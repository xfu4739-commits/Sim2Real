import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss: -alpha_t * (1 - p_t)**gamma * log(p_t).

    ``output`` contains logits and ``target`` contains binary masks of exactly
    the same shape (training uses [batch, time, height, width]). ``p_t`` is
    sigmoid(output) for fire pixels and 1-sigmoid(output) for background.
    If alpha is given, alpha_t=alpha for fire (target=1) and 1-alpha for
    background (target=0). Choose alpha using the training-set class balance;
    it is not estimated independently from each batch. The default alpha=None
    disables class weighting (alpha_t=1 for both classes), preserving the
    previous training default without silently choosing a class balance.
    """

    def __init__(self, alpha=None, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1, or None')
        if gamma < 0:
            raise ValueError('gamma must be non-negative')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unexpected reduction {reduction}')

    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f'output and target shapes must match: {output.shape} != {target.shape}')
        target = target.to(dtype=output.dtype)

        # BCEWithLogits computes -log(p_t) stably, even for large-magnitude logits.
        # exp(-BCE) gives p_t without clamping or detaching the focal factor:
        # gradients must flow through both (1-p_t)**gamma and log(p_t).
        cross_entropy = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        p_t = torch.exp(-cross_entropy)
        loss = (1 - p_t).pow(self.gamma) * cross_entropy
        if self.alpha is not None:
            # Class weighting is applied per pixel: fire gets alpha, background
            # gets 1-alpha. Do not add an axis or mix examples in the batch.
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def test_binary_focal_loss_regression():
    """Check training shape, formula, gradient, alpha and reduction.

    Run from the repository root: python -m lib.focal_loss
    This diagnostic does not change the loss used by training.
    """
    batch, steps, height, width = 2, 3, 4, 5
    # Each sample is confidently correct on its own; an accidental extra axis
    # would mix the two samples and produce a much larger loss.
    logits = torch.empty(batch, steps, height, width)
    logits[0].fill_(3.0)
    logits[1].fill_(-3.0)
    logits.requires_grad_()
    target = torch.zeros_like(logits)
    target[0].fill_(1.0)

    focal = BinaryFocalLoss(gamma=2)
    actual = focal(logits, target)
    prob = torch.sigmoid(logits)
    pt = torch.where(target.bool(), prob, 1 - prob)
    reference = -((1 - pt) ** 2) * torch.log(pt)
    expected = reference.mean()
    print(f"logits/target: {tuple(logits.shape)}")
    print(f"actual loss: {actual.item():.8f}; formula reference: {expected.item():.8f}")
    assert torch.allclose(actual, expected, atol=1e-7)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-8)

    per_pixel = BinaryFocalLoss(reduction='none')(logits, target)
    assert per_pixel.shape == (batch, steps, height, width)
    assert torch.allclose(per_pixel, reference, atol=1e-7)
    assert torch.allclose(BinaryFocalLoss(reduction='sum')(logits, target), reference.sum(), atol=1e-5)
    alpha = 0.75
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    weighted_reference = alpha_t * reference
    weighted_actual = BinaryFocalLoss(alpha=alpha, reduction='none')(logits, target)
    assert torch.allclose(weighted_actual, weighted_reference, atol=1e-7)
    assert torch.allclose(weighted_actual[0], reference[0] * alpha, atol=1e-7)
    assert torch.allclose(weighted_actual[1], reference[1] * (1 - alpha), atol=1e-7)
    try:
        focal(logits, target.unsqueeze(1))
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched shapes must not broadcast')
    print('Confirmed: shape, value, gradient, positive/negative alpha_t and all reductions.')


if __name__ == '__main__':
    test_binary_focal_loss_regression()
