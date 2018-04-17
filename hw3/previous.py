from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pylab
import numpy as np


def init_weights(n_feature, n_class, n_hidden=100):
    # Initialize weights with Standard Normal random variables
    model = dict(
        # Add 1 for bias node, will be connected to each hidden unit
        # this is equivalent to adding a bias per neuron
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )

    return model

# Defines the softmax function.
#For two classes, this is equivalent to the logistic regression
def softmax(x):
    a = max(x)
    return np.exp(x-a) / np.exp(x-a).sum()

# For a single example $x$
def forward(x, model):
    # Input times first layer matrix 
    z_1 = x @ model['W1']

    # ReLU activation goes to second hidden layer
    h = z_1
    h[z_1 < 0] = 0

    # Hidden layer values to output
    hat_y = softmax(h @ model['W2'])

    return h, hat_y

def backward(model, xs, hs, errs):
    """xs, hs1, hs2, errs contain all information 
    (input, hidden state 1, hidden state 2, error)
    of all data in the minibatch""" 
    # errs is the gradient of output layer for the minibatch
    dW2 = (hs.T @ errs)/xs.shape[0]

    # get gradient of last hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = (xs.T @ dh)/xs.shape[0]

    return dict(W1=dW1, W2=dW2)

def get_gradient(model, X_train, y_train, n_class):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

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
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs),
                    np.array(errs))

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
        #print('Iteration (epoch) {}'.format(iter))

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
    # There are only two features in the data X[:,0] and X[:,1]
    n_feature = 2
    # There are only two classes: 0 (purple) and 1 (yellow)
    n_class = 2

    no_iter = 100
    no_runs = 20

    accuracies = np.zeros(no_runs)

    for run in range(no_runs):

        X, y = make_moons(n_samples=3100, noise=0.1)
        X_train, X_dev, y_train, y_dev = \
            train_test_split(X, y, test_size=0.032)

        accuracy = 0
        # Reset model
        model = init_weights(n_feature=n_feature, n_class=n_class)

        # Train the model
        model = gradient_descent(model, X_train, y_train,
                                 n_class, no_iter=no_iter)

        y_pred = np.zeros_like(y_dev)
        for i, x in enumerate(X_dev):
            # Predict the distribution of label
            _, prob = forward(x, model)
            # Get label by picking the most probable one
            y = np.argmax(prob)
            y_pred[i] = y

        # Accuracy of predictions with the true labels and take the percentage
        # Because our dataset is balanced, measuring just the accuracy is OK
        accuracy = (y_pred == y_dev).sum() / y_dev.size
        #print('Accuracy after {} iterations: {}'.format(no_iter,accuracy))
        #pylab.scatter(X_test[:,0], X_test[:,1], c=y_pred)
        #pylab.show()

        accuracies[run] = accuracy

    print(accuracies)
    print('Mean accuracy over test data: {}, std: {}'.format(accuracies.mean(), accuracies.std()))

if __name__ == '__main__':
    main()
