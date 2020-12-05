from os import listdir, rename
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

# List of tuples enumerating (alpha, lambda) values we want to compare for this environment
lambda_val_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
pi_lr_arr = [1e-2]
vf_lr_arr = [1e-3, 1e-2]

for lambda_val in lambda_val_arr:
    for pi_lr in pi_lr_arr:
        for vf_lr in vf_lr_arr:

            plot_comparison = [(0.0, lambda_val, pi_lr, vf_lr), (0.4, lambda_val, pi_lr, vf_lr), (0.8, lambda_val, pi_lr, vf_lr), (0.9, lambda_val, pi_lr, vf_lr), (0.95, lambda_val, pi_lr, vf_lr), (0.99, lambda_val, pi_lr, vf_lr)]
            metric = 'cvar'
            metric = metric.lower()
            all_files = [f for f in listdir('results') if isfile(join('results', f))]

            lower_bound = np.inf
            upper_bound = -np.inf

            plt.figure()

            for f in all_files:
                split_by_underscore = f.split('_')
                if len(split_by_underscore) < 4:
                    print(split_by_underscore)
                a, l, value, policy = float(split_by_underscore[2]), float(split_by_underscore[4]), float(split_by_underscore[6]), float(split_by_underscore[8])
                if (a, l, policy, value) in plot_comparison and metric in f:
                    name = join('results', f)
                    with open(name, 'r') as opened_file:
                        data = opened_file.read().split('\n')
                    data = [float(d) for d in data if len(d) > 0]
                    plt.plot(range(1, len(data) + 1), data, label='Alpha = ' + str(a) + '  Lambda = ' + str(l))
                    lower_bound = min(lower_bound, min(data))
                    upper_bound = max(upper_bound, max(data))

            plt.locator_params(axis='y', nbins=20)
            plt.locator_params(axis='x', nbins=20)
            plt.title('Metric: ' + metric.upper() + ', lambda: ' + str(lambda_val) + ', pi_lr: ' + str(pi_lr) + ', vf_lr: ' + str(vf_lr))
            plt.legend()
            #plt.show()
            plt.savefig('graphs_cvar/' + 'lambda_' + str(lambda_val) + '_policy_' + str(pi_lr) + '_value_' + str(vf_lr) + '.png')
