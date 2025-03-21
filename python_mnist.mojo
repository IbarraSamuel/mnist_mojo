from python import Python, PythonObject
from algorithm.reduction import sum


fn main() raises:
    var np = Python.import_module("numpy")
    data = np.loadtxt("digit-recognizer/train.csv", skiprows=1, delimiter=",")
    test = np.loadtxt("digit-recognizer/test.csv", skiprows=1, delimiter=",")
    m, n = data.shape[0], data.shape[1]
    # np.random.shuffle(data)

    data_train = data.T
    Y_train = data_train[0].astype(np.int32)
    X_train = data_train[1:]
    X_train = X_train / 255.0

    m_train = X_train.shape[1]

    print(Y_train, Y_train.shape)

    W1, b1, W2, b2 = gradient_descent(np, n, X_train, Y_train, 0.10, 500)

    return
    # test_prediction(0, W1, b1, W2, b2)
    # test_prediction(1, W1, b1, W2, b2)
    # test_prediction(2, W1, b1, W2, b2)
    # test_prediction(3, W1, b1, W2, b2)
    # data_test = test.T
    # Y_test = data_test[0]
    # X_test = data_test[1:]
    # X_test = X_test / 255.0

    # dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
    # get_accuracy(dev_predictions, Y_test)


def init_params(
    np: PythonObject,
) -> (PythonObject, PythonObject, PythonObject, PythonObject):
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(np: PythonObject, Z: PythonObject) -> PythonObject:
    return np.maximum(Z, 0)


def softmax(np: PythonObject, Z: PythonObject) -> PythonObject:
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)  # NOTE: Changed to np.sum
    return A


def forward_prop(
    np: PythonObject,
    W1: PythonObject,
    b1: PythonObject,
    W2: PythonObject,
    b2: PythonObject,
    X: PythonObject,
) -> (PythonObject, PythonObject, PythonObject, PythonObject):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(np, Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(np, Z2)
    return Z1, A1, Z2, A2


@always_inline
def ReLU_deriv(Z: PythonObject) -> PythonObject:
    return Z > 0


@always_inline
def one_hot(np: PythonObject, Y: PythonObject) -> PythonObject:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(
    np: PythonObject,
    m: PythonObject,
    Z1: PythonObject,
    A1: PythonObject,
    Z2: PythonObject,
    A2: PythonObject,
    W1: PythonObject,
    W2: PythonObject,
    X: PythonObject,
    Y: PythonObject,
) -> (PythonObject, PythonObject, PythonObject, PythonObject):
    one_hot_Y = one_hot(np, Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(
    W1: PythonObject,
    b1: PythonObject,
    W2: PythonObject,
    b2: PythonObject,
    dW1: PythonObject,
    db1: PythonObject,
    dW2: PythonObject,
    db2: PythonObject,
    alpha: Float32,
) -> (PythonObject, PythonObject, PythonObject, PythonObject):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(np: PythonObject, A2: PythonObject) -> PythonObject:
    return np.argmax(A2, 0)


def get_accuracy(
    np: PythonObject, predictions: PythonObject, Y: PythonObject
) -> PythonObject:
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(
    np: PythonObject,
    n: PythonObject,
    X: PythonObject,
    Y: PythonObject,
    alpha: Float32,
    iterations: Int,
) -> (PythonObject, PythonObject, PythonObject, PythonObject):
    W1, b1, W2, b2 = init_params(np)

    print("Init weights and biases:")
    print("W1: shape: {} and values: \n{}".format(String(W1.shape), String(W1)))
    print("b1: shape: {} and values: \n{}".format(String(b1.shape), String(b1)))
    print("W2: shape: {} and values: \n{}".format(String(W2.shape), String(W2)))
    print("b2: shape: {} and values: \n{}".format(String(b2.shape), String(b2)))
    print("X: shape: {} and values: \n{}".format(String(X.shape), String(X)))
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(np, W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(np, n, Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha
        )
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(np, A2)
            print(get_accuracy(np, predictions, Y))
    return W1, b1, W2, b2


def make_predictions(
    np: PythonObject,
    X: PythonObject,
    W1: PythonObject,
    b1: PythonObject,
    W2: PythonObject,
    b2: PythonObject,
) -> PythonObject:
    _, _, _, A2 = forward_prop(np, W1, b1, W2, b2, X)
    predictions = get_predictions(np, A2)
    return predictions


# def test_prediction(X_train: PythonObject,index: PythonObject, W1: PythonObject, b1: PythonObject, W2: PythonObject, b2: PythonObject) -> PythonObject:
#     current_image = X_train[:, index, None]
#     prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
#     label = Y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)

#     current_image = current_image.reshape((28, 28)) * 255
#     # plt.gray()
#     # plt.imshow(current_image, interpolation='nearest')
#     # plt.show()
