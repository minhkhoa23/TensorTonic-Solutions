import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    bow = np.zeros(len(vocab), dtype = int)
    for i, word in enumerate(vocab):
        bow[i] = tokens.count(word)
    return bow