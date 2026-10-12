"""Use real data's existing six-frame time coordinate for both datasets."""


def normalize_target_timestamps(timestamps):
    """Normalize an ordered source+target sequence to its chronological span.

    Input order carries the task direction: forecast increases in time and
    backtracking decreases. Bounds come from the six selected frames, not from
    the full event, so the two datasets feed comparable positions to the time
    encoder. Only the target half is returned.
    """
    values = list(timestamps)
    if not values or len(values) % 2:
        raise ValueError('timestamps must contain equally many source and target frames')

    first, last = min(values), max(values)
    if first == last:
        normalized = [0.0] * len(values)
    else:
        duration = last - first
        normalized = [float((value - first) / duration) for value in values]
    return normalized[len(values) // 2:]
