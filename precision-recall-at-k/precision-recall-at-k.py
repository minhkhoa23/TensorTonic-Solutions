def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    hits = set(top_k) & set(relevant)
    count_hits = len(hits)
    precision = count_hits / k
    recall = count_hits / len(relevant)
    return [precision, recall]