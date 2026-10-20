"""Aggregate per-target fire-mask diagnostics during validation."""

import torch


def batch_fire_counts(prediction, target, source, reverse):
    """Return [step, tp/fp/fn, growth tp/fp/fn, sample count]."""
    prediction = prediction.bool()
    target = target.bool()
    reference = source[:, -1:].bool()
    if reverse:
        growth_target = reference & ~target
        growth_prediction = reference & ~prediction
    else:
        growth_target = target & ~reference
        growth_prediction = prediction & ~reference
    result = []
    for predicted, truth in ((prediction, target), (growth_prediction, growth_target)):
        result.extend((
            (predicted & truth).sum(dim=(0, 2, 3)),
            (predicted & ~truth).sum(dim=(0, 2, 3)),
            (~predicted & truth).sum(dim=(0, 2, 3)),
        ))
    result.append(torch.full_like(result[0], prediction.shape[0]))
    return torch.stack(result, dim=1).to(torch.float64)


def summarize_fire_counts(counts, epoch):
    rows = []
    for step, values in enumerate(counts.tolist(), 1):
        tp, fp, fn, growth_tp, growth_fp, growth_fn, samples = values
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else float('nan')
        rows.append(dict(
            epoch=epoch, step=step, samples=int(samples),
            iou=ratio(tp, tp + fp + fn),
            precision=ratio(tp, tp + fp),
            recall=ratio(tp, tp + fn),
            area_ratio=ratio(tp + fp, tp + fn),
            mean_fp_pixels=fp / samples if samples else float('nan'),
            mean_fn_pixels=fn / samples if samples else float('nan'),
            growth_iou=ratio(growth_tp, growth_tp + growth_fp + growth_fn),
            growth_recall=ratio(growth_tp, growth_tp + growth_fn),
        ))
    return rows
