import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy import stats

accs_curr = \
    [0.99193548, 1.0, 1.0, 1.0, 1.0,
     1.0, 0.96774194, 1.0, 1.0, 1.0,
     0.99193548, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0]

accs_prev = \
    [0.89, 0.89, 0.9, 0.89, 0.82,
     0.89, 0.87, 0.86, 0.89, 0.93,
     0.93, 0.87, 0.89, 0.91, 0.85,
     0.87, 0.86, 0.88, 0.92, 0.84]


accs_curr = np.array(accs_curr)
accs_prev = np.array(accs_prev)

t_test, p_value = stats.ttest_rel(accs_curr,accs_prev)
print('t_test', t_test, 'p_value', p_value)

print(np.mean(accs_curr), np.std(accs_curr))
print(np.mean(accs_prev), np.std(accs_prev))

data = [accs_curr, accs_prev]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
plt.xticks([1, 2], ['bias+2HL', '1HL'])
fig.savefig('boxplot.png')
