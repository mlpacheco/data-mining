import torch
import numpy as np
import scipy
import torch.nn.functional as F
from sklearn.decomposition import PCA

class CovarianceSVD(torch.nn.Module):

    def __init__(self, n_input, n_components):
        super(CovarianceSVD, self).__init__()
        self.layer = torch.nn.Linear(n_input, n_components)
        self.sigma = torch.nn.Parameter(torch.ones(n_components))

    def forward(self, row, col):
        u = self.layer(row)
        uT = self.layer(col)
        y_hat = torch.sum(self.sigma * u * uT, dim=1)
        return y_hat

    def get_U(self, row, n_rows, n_cols):
        u_params = self.layer(row).data.numpy()
        ridx = [i for i in range(0, n_rows * n_cols, n_rows)]
        return u_params[ridx,:]

    def get_sigma(self):
        diag = self.sigma.data
        sigma = torch.diag(diag)
        return sigma.numpy()

def get_one_hot(idx, dim):
    onehot = [0.0] * dim
    onehot[idx] = 1.0
    return onehot

def get_full_batch(XTX, n_rows, n_cols):
    Mrow = []; Mcol = []; Y = []
    for row in range(n_rows):
        for col in range(n_cols):
            row_repr = get_one_hot(row, n_rows)
            col_repr = get_one_hot(col, n_cols)
            Mrow.append(row_repr)
            Mcol.append(col_repr)

            Y.append(float(XTX[row][col]))
    return Mrow, Mcol, Y

def scale(arr, min_, max_):
    arr += -(np.min(min_))
    arr /= np.max(arr) / (max_ - min_)
    arr += min_
    return arr

def main():
    # hyper-parameters
    lrate = 1e-1
    num_epochs = 5000

    # X dimensions: N = 1000, D = 100
    X = np.loadtxt("PCAdata.txt", delimiter=',', dtype=np.float32)
    XTX = np.matmul(X.T, X)

    n_rows = XTX.shape[0]
    n_cols = XTX.shape[1]
    n_components = 10

    # n_rows == n_cols
    model = CovarianceSVD(n_rows, n_components)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    Mrow, Mcol, Y = get_full_batch(XTX, n_rows, n_cols)
    Mrow = torch.FloatTensor(Mrow)
    Mcol = torch.FloatTensor(Mcol)
    Y = torch.FloatTensor(Y)

    Mrow = torch.autograd.Variable(Mrow).float()
    Mcol = torch.autograd.Variable(Mcol).float()
    Y = torch.autograd.Variable(Y).float()

    for epoch in range(num_epochs):
        Y_hat = model(Mrow, Mcol)
        loss = loss_fn(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Loss", loss.data[0])
    Unn = model.get_U(Mrow, n_rows, n_cols)
    Snn = model.get_sigma()
    Snn_invsqrt = scipy.linalg.sqrtm(scipy.linalg.inv(Snn))
    Znn = np.matmul(np.matmul(X, Unn), Snn_invsqrt)

    # Now compare with PCA
    pca = PCA(n_components=10)
    Zsk = pca.fit_transform(X)

    # Report Frobinious Norm
    Fnorm = scipy.linalg.norm(Znn - Zsk)
    print("Frobinous Norm Znn - Zsk", Fnorm)

    # Rescale Znn and print again
    max_, min_ = np.max(Zsk), np.min(Zsk)
    Znn_scaled = scale(Znn, min_, max_)
    Fnorm = scipy.linalg.norm(Znn_scaled - Zsk)
    print("Frobinous Norm Znn_scaled - Zsk", Fnorm)

if __name__ == "__main__":
    main()
