import time
from sklearn import metrics, svm, datasets, utils, model_selection
from helpers import bitmap_handler


def linear_svm(test_size: float = .2):
    """
    Uses SVM with a linear kernel to predict the MNIST set. LinearSVC uses the one-vs-rest multi-class strategy.

    :param test_size: float
        Scale of test size used in this case.
    """
    start_time = time.time()

    mnist = datasets.fetch_mldata('MNIST original', data_home='./data')
    print("Fetched %d rows." % len(mnist.target))

    print("Shuffle data set")
    mnist.data, mnist.target = utils.shuffle(mnist.data, mnist.target)

    print("Split into training set and test/validation set (test_size: %.1f %%)" % (test_size * 100))
    x_train, x_test, y_train, y_test = model_selection.train_test_split(mnist.data, mnist.target, test_size=test_size)

    print("Training SVM...")
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)

    print("Making predictions...")
    prediction = clf.predict(x_test)

    print("Evaluating results...")
    classification_report = metrics.classification_report(y_test, prediction)
    confusion_matrix = metrics.confusion_matrix(y_test, prediction)

    print('Classification report:\n%s\n\nConfusion matrix:\n%s\n'
          % (classification_report, confusion_matrix))

    m, s = divmod(time.time() - start_time, 60)
    print("Overall running time: %d min. %d sec." % (m, s))

    bitmap_handler.compare_wrong_results(x_test, prediction, y_test)
