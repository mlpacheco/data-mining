import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(title, train_sizes, train_scores, test_scores):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g", label="Development score")

    plt.legend(loc="best")
    return plt



title = "Learning Curves"

train_sizes = [100, 500, 1000, 2000, 3000]
train_scores = [1.0, 1.0, 0.999, 0.999, 0.999]
dev_scores = [0.994, 0.997, 0.998, 0.998, 0.998]


plot_learning_curve(title, train_sizes, train_scores, dev_scores)
plt.show()

