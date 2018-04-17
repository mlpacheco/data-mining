import numpy as np
import math
import sys

X = [(0,0),(0,1),(0,2),
     (1,0),(1,1),(1,2),
     (2,0),(2,1),(2,2),
     (3,0),(3,1),(3,2),
     (4,0),(4,1),(4,2),
     (5,0),(5,1),(5,2)]

y = [-1,-1,-1,-1,1,-1,-1,-1,-1,
     1,1,1,1,1,1,1,1,1]

d = [1/18.0] * 18
alphas = []; preds = []

K = int(sys.argv[1])
for k in range(K):
    print "d", d
    if k == 3:
        y_hat = [1 if x_2 < 2 else -1 for (x_1, x_2) in X]
    elif k == 2:
        y_hat = [1 if x_2 > 0 else -1 for (x_1, x_2) in X]
    elif k == 1:
        y_hat = [-1 if x_1 < 1 else 1 for (x_1, x_2) in X]
    elif k == 0:
       y_hat = [-1 if x_1 <= 2 else 1 for (x_1, x_2) in X]

    print y_hat

    e_k = sum([d[i]*1.0 if y_i != yh_i else 0.0 for i, (y_i, yh_i) in enumerate(zip(y, y_hat))])
    alpha_k = 1.0/2 * math.log((1-e_k)/e_k)
    print "alpha_k", alpha_k
    alphas.append(alpha_k)
    preds.append(y_hat)

    d_nums = []
    for i, d_i in enumerate(d):
        d_nums.append(d_i * math.exp(-alpha_k * y[i] * y_hat[i]))
    Z = sum(d_nums)
    d = (np.asarray(d_nums) / Z)

predictions = []
print "alphas", alphas

#alphas = [1.416606672028108, 0.589327498170823, 0.9052860137994106]
for i in range(len(X)):
    sum = 0.0
    for k in range(K):
        sum += alphas[k] * preds[k][i]
    if sum >= 0:
        predictions.append(1)
    else:
        predictions.append(-1)
print
print y
print predictions
