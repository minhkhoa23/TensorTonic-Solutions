import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)

    label, count = np.unique(y_train, return_counts=True)

    majority = label[np.argmax(count)]

    prediction = np.full(len(X_test), majority)

    return prediction