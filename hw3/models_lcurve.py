from nn_model import *
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier

def main():
    train_x, train_y = read_train_data("data/train.csv")
    test_x = read_test_data_x("data/test.csv")
    test_y = read_test_data_y("data/pred.csv")

    train_x[np.isnan(train_x)] = 0
    test_x[np.isnan(test_x)] = 0

    # normalizing data
    train_x, mean, std = normalize(train_x)
    test_x = normalize_test(test_x, mean, std)

    aucs_test = []
    aucs_train = []

    proportions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for prop in proportions:
        size = int(train_x.shape[0]*prop)


        # Uncomment for logistic regression
        # clf = linear_model.LogisticRegression(penalty='l2', C=1e4)

        # Uncomment for GBDT
        clf = GradientBoostingClassifier(n_estimators=100,
                                         learning_rate=1.0,
                                         max_depth=2)

        clf.fit(train_x[:size], train_y[:size])
        pred_test_y = clf.predict(test_x)
        pred_train_y = clf.predict(train_x[:size])


        # Uncomment for NNet
        '''
        model = create_model(idim, odim, hdim1, hdim2) # creating model structure
        trained_model = nn_train(train_x_fold, train_y_fold, model, 100, 0.01, 100) # training model
        pred_test_y = nn_predict(test_x, trained_model)
        pred_train_y = nn_predict(train_x[:size], trained_model)
        '''

        # AUC for test data
        fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_test_y, pos_label=1)
        aucs_test.append(metrics.auc(fpr, tpr))

        # AUC for train data
        fpr, tpr, thresholds = metrics.roc_curve(train_y[:size], pred_train_y, pos_label=1)
        aucs_train.append(metrics.auc(fpr, tpr))

    print(aucs_train)
    print(aucs_test)

    #title = "Learning Curves (Logistic Regression with L2)"
    title = "Learning Curves (Gradient Boosting Decision Trees)"
    #title = "Learning Curves (NNets)"
    plot_learning_curve(title, proportions, aucs_train, aucs_test)
    plt.show()


if __name__ == '__main__':
    main()
