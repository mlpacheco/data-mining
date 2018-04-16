import torch
import torch.autograd as autograd
import numpy as np
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# This method creates a pytorch model
def create_model(idim, odim, hdim1, hdim2):
    model = torch.nn.Sequential(
            torch.nn.Linear(idim, hdim1),
            torch.nn.BatchNorm1d(hdim1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hdim1, hdim2),
            torch.nn.BatchNorm1d(hdim1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hdim2, odim),
            torch.nn.LogSoftmax()
            )
    return model

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(targets))

    end_idx = max(len(targets) - batchsize + 1, 1)

    for start_idx in range(0, end_idx, batchsize):
        # print batchsize
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield [inputs[i] for i in excerpt],\
              [int(targets[i]) for i in excerpt]


# This method trains a model with the given data
# Epoch is the number of training iterations
# lrate is the learning rate
def nn_train(train_x, train_y, model, epoch, lrate, batchsize):
    inputs = []


    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    for itr in range(epoch):

        for batched_x, batched_y in \
                iterate_minibatches(train_x, train_y, batchsize,
                                    shuffle=True):
            X = autograd.Variable(torch.FloatTensor(batched_x),
                                  requires_grad=True)
            Y = autograd.Variable(torch.LongTensor(batched_y),
                                  requires_grad=False)


            y_pred = model(X)

            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#        print("Epoch: {}  Acc: {}".format(itr,nn_test(test_x,test_y,model)))

    return model

# Pass the trained model along with test data to this method to get accuracy
# The method return accuracy value
def nn_test(test_x, test_y, model):

    X = autograd.Variable(torch.from_numpy(test_x), requires_grad=False).float()
    y_pred = model(X)
    _ , idx = torch.max(y_pred, 1)

#    test_y = test_y.values[:,0]
    return (1.*np.count_nonzero((idx.data.numpy() == test_y).astype('uint8')))/len(test_y)

def nn_predict(test_x, model):
    X = autograd.Variable(torch.from_numpy(test_x), requires_grad=False).float()
    y_pred = model(X)
    _ , idx = torch.max(y_pred, 1)
    return idx.data.numpy()


def normalize(data_list):
    # compute normalization parameter
    utter = np.concatenate(data_list, axis=0)
    mean  = np.mean(utter)
    utter -= mean
    std   = np.std(utter)
    utter /= std

    # normalize data
    for data in data_list:
        data -= mean
        data /= std

    return data_list, mean, std

def normalize_test(data_list, mean, std):
    for data in data_list:
        data -= mean
        data /= std

    return data_list

# Method reads training data and drops loan id field
def read_train_data(filename):
    f = open(filename, "r")
    x = []
    y = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        y.append(line[len(line)-1])
        line = line[1:len(line)-1]
        x.append(line)


    x = np.array(x)
    x = x.astype(np.float)

    y = np.array(y)
    y = y.astype(np.int)
    return (x,y)



# Method reads features of test data
def read_test_data_x(filename):
    f = open(filename, "r")
    x = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        line = line[1:len(line)-1]
        x.append(line)


    x = np.array(x)
    x = x.astype(np.float)

    return x

# Method reads labels of test data
def read_test_data_y(filename):
    f = open(filename, "r")
    y = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        y.append(line[len(line)-1])



    y = np.array(y)
    y = y.astype(np.int)
    return y

def plot_learning_curve(title, train_sizes, train_scores, test_scores):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g", label="Development score")

    plt.legend(loc="best")

def main():

    train_x, train_y = read_train_data("data/train.csv")
    test_x = read_test_data_x("data/test.csv")
    test_y = read_test_data_y("data/pred.csv")

    train_x[np.isnan(train_x)] = 0
    test_x[np.isnan(test_x)] = 0
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_x, mean, std = normalize(train_x)
    test_x = normalize_test(test_x, mean, std)


    idim = 25  # input dimension
    hdim1 = 64 # hidden layer one dimension
    hdim2 = 64 # hidden layer two dimension
    odim = 2   # output dimension


    aucs_nn = [];
    aucs_lr = [];
    aucs_gbdt= [];

    fold_size = int(round(train_x.shape[0]/10))

    config_folds = []

    for fold, i in enumerate(range(0, train_x.shape[0], fold_size)):

        test_x_fold = train_x[i:i+fold_size]
        test_y_fold = train_y[i:i+fold_size]

        train_x_fold = np.vstack((train_x[0:i], train_x[i+fold_size:train_x.shape[0]]))
        train_y_fold = np.hstack((train_y[0:i], train_y[i+fold_size:train_x.shape[0]]))

        aucs_grid = []
        config_grid = []

        for learning_rate in [10e-2, 10e-3]:
            for batch_size in [100, 1000]:

                config_grid.append({'lr': learning_rate,
                               'batch_size': batch_size})

                train_x_curr, dev_x,\
                train_y_curr, dev_y =  train_test_split(train_x_fold, train_y_fold,
                                                        test_size=0.2)

                model = create_model(idim, odim, hdim1, hdim2) # creating model structure
                trained_model = nn_train(train_x_curr, train_y_curr,
                                         model, 100, learning_rate, batch_size) # training model

                pred_nn_y = nn_predict(dev_x, trained_model)
                fpr, tpr, thresholds = metrics.roc_curve(
                        dev_y, pred_nn_y, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                aucs_grid.append(auc)

        # Select best model for fold
        best_config = np.argmax(aucs_grid)
        print("Fold", fold, "Best parameters", config_grid[best_config])
        config_folds.append(config_grid[best_config])

    # Uncoment to compare new model with other models
    '''
    print("Testing models")
    for fold, i in enumerate(range(0, train_x.shape[0], fold_size)):

        learning_rate = config_folds[fold]['lr']
        batch_size = config_folds[fold]['batch_size']

        test_x_fold = train_x[i:i+fold_size]
        test_y_fold = train_y[i:i+fold_size]

        train_x_fold = np.vstack((train_x[0:i], train_x[i+fold_size:train_x.shape[0]]))
        train_y_fold = np.hstack((train_y[0:i], train_y[i+fold_size:train_x.shape[0]]))

        # Neural Network
        model = create_model(idim, odim, hdim1, hdim2) # creating model structure
        trained_model = nn_train(train_x_fold, train_y_fold,
                                 model, 100, learning_rate, batch_size) # training model
        pred_nn_y = nn_predict(test_x_fold, trained_model)

        fpr, tpr, thresholds = metrics.roc_curve(test_y_fold, pred_nn_y, pos_label=1)
        aucs_nn.append(metrics.auc(fpr, tpr))

        # Logistic Regression
        lrg = linear_model.LogisticRegression(penalty='l2', C=1e4)
        lrg.fit(train_x_fold, train_y_fold)
        pred_lrg_y = lrg.predict(test_x_fold)

        fpr, tpr, thresholds = metrics.roc_curve(test_y_fold, pred_lrg_y, pos_label=1)
        aucs_lr.append(metrics.auc(fpr, tpr))


        # Gradient Boosted Decision Trees
        gbdt = GradientBoostingClassifier(n_estimators=100,
                                         learning_rate=1.0,
                                         max_depth=2)
        gbdt.fit(train_x_fold, train_y_fold)
        pred_gbdt_y = gbdt.predict(test_x_fold)

        fpr, tpr, thresholds = metrics.roc_curve(test_y_fold, pred_gbdt_y, pos_label=
        1)
        aucs_gbdt.append(metrics.auc(fpr, tpr))


    print("Mean AUC LR", np.mean(aucs_lr))
    print("Mean AUC GBDT", np.mean(aucs_gbdt))
    print("Mean AUC NNet", np.mean(aucs_nn))

    t_test, p_value = stats.ttest_rel(aucs_lr, aucs_nn)
    print("LogReg vs. NNet", t_test, p_value)

    t_test, p_value = stats.ttest_rel(aucs_gbdt, aucs_nn)
    print("GBDT vs. NNet", t_test, p_value)
    '''


if __name__ == "__main__":
    main()
