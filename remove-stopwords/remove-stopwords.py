def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    result = []
    for i in tokens:
        if i not in stopwords:
            result.append(i)
    return result