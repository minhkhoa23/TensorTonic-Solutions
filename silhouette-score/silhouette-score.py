import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = X.shape[0]
    unique_labels = np.unique(labels)

    scores = []

    for i in range(n):

        # ----- Calculate a(i) -----
        same_cluster = np.where(labels == labels[i])[0]
        same_cluster = same_cluster[same_cluster != i]

        # Singleton cluster
        if len(same_cluster) == 0:
            scores.append(0.0)
            continue

        distances = np.linalg.norm(
            X[same_cluster] - X[i],
            axis=1
        )

        a = np.mean(distances)

        # ----- Calculate b(i) -----
        b = np.inf

        for label in unique_labels:

            if label == labels[i]:
                continue

            other_cluster = X[labels == label]

            distances = np.linalg.norm(
                other_cluster - X[i],
                axis=1
            )

            mean_distance = np.mean(distances)

            b = min(b, mean_distance)

        # ----- Silhouette score -----
        if max(a, b) == 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)

        scores.append(s)

    return float(np.mean(scores))