from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pylab
import numpy as np


def init_weights(n_feature, n_class, n_hidden=100):
    # Initialize weights with Standard Normal random variables
    model = dict(
        # Add 1 for bias node, will be connected to each hidden unit
        # this is equivalent to adding a bias per neuron
        W1=np.random.randn(n_feature + 1, n_hidden),
        # Add one extra fully connected hidden layer
        W2=np.random.randn(n_hidden, n_hidden),
        W3=np.random.randn(n_hidden, n_class)
    )

    return model

# Defines the softmax function.
#For two classes, this is equivalent to the logistic regression
def softmax(x):
    return np.exp(x) / np.exp(x).sum()

# For a single example $x$
def forward(x, model):
    # Input times first layer matrix 
    z_1 = x @ model['W1']

    # ReLU activation goes to hidden layer
    h_1 = z_1
    h_1[z_1 < 0] = 0

    # Hidden units times second layer matrix
    z_2 = h_1 @ model['W2']

    # ReLU activation goes to second hidden layer
    h_2 = z_2
    h_2[z_2 < 0] = 0

    # Hidden layer values to output
    hat_y = softmax(h_2 @ model['W3'])

    return h_1, h_2, hat_y

def backward(model, xs, hs1, hs2, errs):
    """xs, hs1, hs2, errs contain all information 
    (input, hidden state 1, hidden state 2, error)
    of all data in the minibatch""" 
    # errs is the gradient of output layer for the minibatch
    dW3 = (hs2.T @ errs)/xs.shape[0]

    # get gradient of last hidden layer
    dh2 = errs @ model['W3'].T
    dh2[hs2 <= 0] = 0

    dW2 = (hs1.T @ dh2)/xs.shape[0]

    dh1 = dh2 @ model['W2'].T
    dh1[hs1 <= 0] = 0

    dW1 = (xs.T @ dh1)/xs.shape[0]

    return dict(W1=dW1, W2=dW2, W3=dW3)

def get_gradient(model, X_train, y_train, n_class):
    xs, hs1, hs2, errs = [], [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h_1, h_2, y_pred = forward(x, model)

        # Create one-hot coding of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of the examples
        # x: input
        # h_1: hidden state for first hidden layer
        # h_2: hidden state for second hidden layer
        # err: gradient of output layer
        xs.append(x)
        hs1.append(h_1)
        hs2.append(h_2)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs1),
                    np.array(hs2), np.array(errs))

def gradient_step(model, X_train, y_train, n_class, learning_rate = 1e-1):
    grad = get_gradient(model, X_train, y_train, n_class)
    model = model.copy()

    # Update every parameters in our networks 
    #(W1, W2 and W3) using their gradients
    for layer in grad:
        # Careful, learning rate should depend on mini-batch size
        model[layer] += learning_rate * grad[layer]

    return model

def gradient_descent(model, X_train, y_train, n_class, no_iter=10):

    minibatch_size = 50

    for iter in range(no_iter):
        print('Iteration (epoch) {}'.format(iter))

        ## MINI-BATCH: Shuffles the training data to sample without replacement
        indices = list(range(0,X_train.shape[0]))
        np.random.shuffle(indices)
        X_train = X_train[indices,:]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current mini-batch
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            model = gradient_step(model, X_train_mini, y_train_mini,
                                  n_class, learning_rate = 1e-1)

    return model

def main():
    X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, random_state=42,test_size=0.3)

    # Create bias nodes
    bias_tr = np.ones((X_train.shape[0], 1))
    bias_tst = np.ones((X_test.shape[0], 1))
    # Add bias nodes to input
    X_train = np.hstack((X_train, bias_tr))
    X_test = np.hstack((X_test, bias_tst))

    # There are only two features in the data X[:,0] and X[:,1]
    n_feature = 2
    # There are only two classes: 0 (purple) and 1 (yellow)
    n_class = 2

    no_iter = 10

    # Reset model
    model = init_weights(n_feature=n_feature, n_class=n_class)

    # Train the model
    model = gradient_descent(model, X_train, y_train,
                             n_class, no_iter=no_iter)

    y_pred = np.zeros_like(y_test)

    accuracy = 0

    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, _, prob = forward(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

    # Accuracy of predictions with the true labels and take the percentage
    # Because our dataset is balanced, measuring just the accuracy is OK
    accuracy = (y_pred == y_test).sum() / y_test.size
    print('Accuracy after {} iterations: {}'.format(no_iter,accuracy))
    #pylab.scatter(X_test[:,0], X_test[:,1], c=y_pred)
    #pylab.show()


if __name__ == '__main__':
    main()
