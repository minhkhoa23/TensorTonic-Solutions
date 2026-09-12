import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    # Write code here
    N = len(docs)

    if N == 0:
        return np.array([])

    avgdl = sum(len(doc) for doc in docs) / N

    query_terms = set(query_tokens)

    df = {
        term: sum(term in doc for doc in docs)
        for term in query_terms
    }

    scores = []

    for doc in docs:
        tf = Counter(doc)
        score = 0.0

        for term in query_terms:
            freq = tf[term]

            if freq == 0:
                continue

            idf = math.log(
                (N - df[term] + 0.5) /
                (df[term] + 0.5)
                + 1
            )

            denom = (
                freq
                + k1 * (
                    1 - b
                    + b * len(doc) / avgdl
                )
            )

            score += idf * freq * (k1 + 1) / denom

        scores.append(score)

    return np.array(scores)