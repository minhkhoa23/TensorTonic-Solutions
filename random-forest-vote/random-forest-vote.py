import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.array(predictions)

    n_samples = predictions.shape[1]
    result = []

    for i in range(n_samples):
        counts = {}

        for tree in range(predictions.shape[0]):
            label = predictions[tree][i]
            counts[label] = counts.get(label, 0) + 1

        max_count = max(counts.values())

        winner = min(
            label for label, count in counts.items()
            if count == max_count
        )

        result.append(winner)

    return result