from os import listdir, rename
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

# List of tuples enumerating (alpha, lambda) values we want to compare for this environment
#lambda_val_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


plot_comparison = [(0.95, 0), (0.95, 0.2), (0.95, 0.4), (0.95, 0.6), (0.95, 0.8), (0.95, 1)]
metric = 'ret_wc_ratio'
metric = metric.lower()
all_files = [f for f in listdir('results') if isfile(join('results', f))]

lower_bound = np.inf
upper_bound = -np.inf

plt.figure()
# for below colors = [red, orange, yellow, green, blue, indigo, violet]
colors = ['#FF0000', '#ffa500', '#ffff00', '#00FF00', '#0000FF', '#4B0082', '#8F00FF']
count = 0
for i in range(len(plot_comparison)):
    for f in all_files:
        split_by_underscore = f.split('_')
        a, l = float(split_by_underscore[3]), float(split_by_underscore[5])
        if l == plot_comparison[i][1] and metric in f:
            name = join('results', f)
            with open(name, 'r') as opened_file:
                data = opened_file.read().split('\n')
            data = [float(d) for d in data if len(d) > 0]
            plt.plot(range(1, len(data) + 1), data, label='Lambda = ' + str(l), color=colors[count])
            count += 1
            lower_bound = min(lower_bound, min(data))
            upper_bound = max(upper_bound, max(data))

plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis='x', nbins=20)
plt.title('Ratio of True Return and Worst Case Return ' + ', Alpha: ' + str(0.95))
plt.legend()
#plt.show()
plt.savefig('graphs/' + 'alpha_' + str(0.95) + '.png')
