import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).tokenized_docs = [
        doc.lower().split()
        for doc in documents
    ]
    """
    # Write code here
    tokenized_docs = [doc.lower().split() for doc in documents]

    vocabulary = sorted(set(word for tokens in tokenized_docs for word in tokens))
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}

    n_docs = len(documents)
    tfidf_matrix = np.zeros((n_docs, len(vocabulary)))

    df = Counter()
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[word] += 1

    for i, tokens in enumerate(tokenized_docs):
        tf = Counter(tokens)

        for word, count in tf.items():
            j = word_to_idx[word]
            idf = math.log(n_docs / df[word])
            tfidf_matrix[i, j] = (count / len(tokens)) * idf

    return tfidf_matrix, vocabulary