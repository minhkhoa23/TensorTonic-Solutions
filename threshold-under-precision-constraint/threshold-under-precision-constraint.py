def choose_threshold(y_true: list, scores: list, min_precision: float) -> dict:
    """
    Returns the threshold with the highest recall whose precision is at least min_precision.
    """
    if len(y_true) != len(scores):
        raise ValueError("Input lengths must match")

    if not 0 <= min_precision <= 1:
        raise ValueError("Invalid min_precision")

    if not y_true or sum(y_true) == 0:
        return {"threshold": None, "precision": None, "recall": None}

    pairs = sorted(zip(scores, y_true), reverse=True)

    total_positive = sum(y_true)
    tp = 0
    fp = 0

    best_threshold = None
    best_precision = -1.0
    best_recall = -1.0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1

        # Process all samples with the same score
        if i + 1 < len(pairs) and pairs[i + 1][0] == score:
            continue

        precision = tp / (tp + fp)
        recall = tp / total_positive

        if precision >= min_precision:
            if (recall > best_recall or
                (recall == best_recall and precision > best_precision)):

                best_threshold = score
                best_precision = precision
                best_recall = recall

    if best_threshold is None:
        return {
            "threshold": None,
            "precision": None,
            "recall": None
        }

    return {
        "threshold": float(best_threshold),
        "precision": best_precision,
        "recall": best_recall
    }
