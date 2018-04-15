import torch
import torch.autograd as autograd
import numpy as np
import torch.optim as optim
from sklearn import metrics
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

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
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g", label="Development score")

    plt.legend(loc="best")
    return fig

def main():

    # You are provided a skelton pytorch nn code for your experiments
    # Use this main code to understand how the methods are used
    # There are two methods nn_train, which is used to train a model (created using create_model method)
    # Other than data nn train takes learning rate and number of iterations as parameters. 
    # Please change the given values for your experiments.
    # nn_train returns a trained model
    # pass this model to nn_test to get accuracy
    # nn_test return 0/1 accuracy 
    # Please read pytorch documentation if you need more details on implementing nn using pytorch
    # training and test data is read as numpy arrays
    train_x, train_y = read_train_data("train.csv")
    test_x = read_test_data_x("test.csv")
    test_y = read_test_data_y("pred.csv")

  # You can use python pandas if you want to fasten reading file , the code is commented
  #  df = pd.read_csv("train.csv")
  #  train_x = df.drop(columns = ['Loan ID','Status (Fully Paid=1, Not Paid=0)'], axis = 1)
  #  train_y = df['Status (Fully Paid=1, Not Paid=0)']
  #  train_x = train_x.values
  #  train_y = train_y.values

  #  df = pd.read_csv("test.csv")
  #  test_x = df.drop(columns = ['Loan ID','Status (Fully Paid=1, Not Paid=0)'], axis = 1)
  #  test_x = test_x.values
  #  df = pd.read_csv("pred.csv")
  #  test_y = df.drop(columns = ['Loan ID'], axis = 1)

    train_x[np.isnan(train_x)] = 0
    test_x[np.isnan(test_x)] = 0
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_x, mean, std = normalize(train_x)
    test_x = normalize_test(test_x, mean, std)


    idim = 25  # input dimension
    hdim1 = 64 # hidden layer one dimension
    hdim2 = 64 # hidden layer two dimension
    odim = 2   # output dimension


    aucs_test = []
    aucs_train = []

    proportions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for prop in proportions:
        size = int(train_x.shape[0]*prop)
        model = create_model(idim, odim, hdim1, hdim2) # creating model structure
        trained_model = nn_train(train_x[:size], train_y[:size], model, 100, 0.01, 100) # training model
        #print(nn_test(test_x, test_y, trained_model)) # testing model

        pred_test_y = nn_predict(test_x, trained_model)
        pred_train_y = nn_predict(train_x[:size], trained_model)


        # AUC for test data
        fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_test_y, pos_label=1)
        aucs_test.append(metrics.auc(fpr, tpr))

        # AUC for train data
        fpr, tpr, thresholds = metrics.roc_curve(train_y[:size], pred_train_y, pos_label=1)
        aucs_train.append(metrics.auc(fpr, tpr))


    title = "Learning Curves"
    fig = plot_learning_curve(title, proportions, aucs_train, aucs_test)
    fig.savefig('auc_lcurves.png')


if __name__ == "__main__":
    main()
