import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if max_len == None:
        L = max(len(seq) for seq in seqs) if seqs else 0
    else:
        L = max_len
    result = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        result[i, :length] = seq[:length]

    return result