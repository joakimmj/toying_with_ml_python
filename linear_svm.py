from sklearn import metrics, svm, datasets, utils, model_selection
import time
import bitmap_handler


def run(test_size: float = .2):
    start_time = time.time()

    mnist = datasets.fetch_mldata('MNIST original', data_home='./')
    print("Fetched %d rows." % len(mnist.target))

    print("Shuffle data set")
    mnist.data, mnist.target = utils.shuffle(mnist.data, mnist.target)

    print(mnist.data[0])
    print(mnist.target[0])

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

    bitmap_handler.compare(x_test, prediction, y_test)


if __name__ == "__main__":
    run()
