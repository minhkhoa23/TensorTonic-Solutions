def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    unique_items = set()

    for user_recs in recommendations:
        for item in user_recs:
            unique_items.add(item)

    return len(unique_items) / n_items