import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    if k <= 0 or len(relevance_scores) == 0:
        return 0.0

    k = min(k, len(relevance_scores))

    dcg = 0.0

    for i in range(k):
        rel = relevance_scores[i]

        gain = 2 ** rel - 1
        discount = math.log2(i + 2)

        dcg += gain / discount

    ideal_scores = sorted(relevance_scores, reverse = True)

    idcg = 0.0

    for i in range(k):
        rel = ideal_scores[i]

        gain = 2 ** rel - 1
        discount = math.log2(i + 2)

        idcg += gain / discount

    if idcg == 0:
        return 0.0

    return dcg / idcg

relevance_scores = [3, 2, 3, 0, 1]
k = 3

print(ndcg(relevance_scores, k))