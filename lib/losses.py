"""Shared loss composition for training and checkpoint evaluation."""


def combined_losses(prediction, target, dice, focal):
    """Return total, Dice and Focal tensors without detaching their gradients."""
    dice_value = dice(prediction, target, use_sigmoid=True)
    focal_value = focal(prediction, target)
    return dice_value + focal_value, dice_value, focal_value
