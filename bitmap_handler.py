import numpy as np
import matplotlib.pyplot as plt


def print_bitmap(bitmap, width: int = 28, height: int = 28):
    rows = np.empty(shape=(0, width))

    for h in range(height):
        rows = np.append(rows, [bitmap[h * width: (h + 1) * width]], axis=0)

    plt.imshow(rows)
    plt.gray()
    plt.show()


def compare(img, prediction, labels):
    for i in range(len(labels)):
        if prediction[i] != labels[i]:
            plt.title("Label: %d, Prediction: %d" % (labels[i], prediction[i]))
            print_bitmap(img[i])
