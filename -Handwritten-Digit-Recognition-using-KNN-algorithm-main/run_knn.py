import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn(X, Y, query_point, k=5):
    vals = []
    m = X.shape[0]

    for i in range(m):
        d = euclidean_distance(query_point, X[i])
        vals.append((d, Y[i]))

    vals = sorted(vals, key=lambda x: x[0])
    vals = vals[:k]

    # extract labels from nearest neighbors and pick the most frequent
    labels = [v[1] for v in vals]
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    index = counts.argmax()
    pred = unique[index]
    return pred


def calculate_accuracy(X_test, Y_test, X_train, Y_train, k=5):
    predictions = []
    for test_point in X_test:
        pred_label = knn(X_train, Y_train, test_point, k=k)
        predictions.append(pred_label)
    predictions = np.array(predictions)
    accuracy = (predictions == Y_test).sum() / Y_test.shape[0]
    return accuracy


if __name__ == '__main__':
    # load a small digits dataset (8x8 images) so this script runs without external CSVs
    digits = load_digits()
    X = digits.data
    Y = digits.target

    # normalize pixel values to [0,1]
    X = X / X.max()

    # split 80/20
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # limit to first 50 test samples to keep runtime low
    X_test_small = X_test[:50]
    Y_test_small = Y_test[:50]

    print('Training samples:', X_train.shape[0])
    print('Test samples (used):', X_test_small.shape[0])

    accuracy = calculate_accuracy(X_test_small, Y_test_small, X_train, Y_train, k=5)
    print('Accuracy (first 50 test samples):', accuracy)

    # save an example test image to file
    example = X_test_small[3]
    img = example.reshape((8, 8))  # digits dataset images are 8x8
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    out_path = 'example_digit.png'
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    print('Saved example image to', out_path)
