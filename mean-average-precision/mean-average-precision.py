import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    """
    Returns a dictionary with map_value and ap_per_query.
    """
    # Write code here
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Sắp xếp theo score giảm dần
        order = np.argsort(y_score)[::-1]
        sorted_true = y_true[order]

        # Chỉ xét top-k
        if k is not None:
            sorted_true = sorted_true[:k]

        relevant_found = 0
        precision_sum = 0.0

        for rank, relevant in enumerate(sorted_true, start=1):
            if relevant == 1:
                relevant_found += 1
                precision_sum += relevant_found / rank

        # Tổng relevant phải lấy từ y_true ban đầu
        total_relevant = np.sum(y_true)

        if total_relevant == 0:
            ap = 0.0
        else:
            ap = precision_sum / total_relevant

        ap_per_query.append(float(ap))

    map_value = np.mean(ap_per_query) if ap_per_query else 0.0

    return {
        "map_value": float(map_value),
        "ap_per_query": ap_per_query
    }