from collections import Counter

def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
     # Step 1: Convert corpus to UTF-8 byte IDs
    sequences = [
        list(text.encode("utf-8"))
        for text in corpus
    ]

    # Step 2: Initialize 256 base tokens
    token_bytes = {
        i: bytes([i])
        for i in range(256)
    }

    vocab = []
    merges = []

    # Step 3: Learn BPE merges
    while 256 + len(merges) < vocab_size:

        pair_freq = Counter()

        # Count adjacent pairs
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freq[pair] += 1

        if not pair_freq:
            break

        # Choose most frequent pair
        # Tie: lexicographically largest byte sequence
        best_pair = max(
            pair_freq,
            key=lambda p: (
                pair_freq[p],
                token_bytes[p[0]],
                token_bytes[p[1]]
            )
        )

        # Step 4: Create new token
        new_id = 256 + len(merges)

        new_bytes = (
            token_bytes[best_pair[0]]
            + token_bytes[best_pair[1]]
        )

        token_bytes[new_id] = new_bytes

        vocab.append([new_id, list(new_bytes)])

        merges.append([
            best_pair[0],
            best_pair[1],
            new_id
        ])

        # Step 5: Replace all matching pairs
        new_sequences = []

        for seq in sequences:
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i + 1 < len(seq)
                    and seq[i] == best_pair[0]
                    and seq[i + 1] == best_pair[1]
                ):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append(new_seq)

        sequences = new_sequences

    return {
        "vocab": vocab,
        "merges": merges
    }
