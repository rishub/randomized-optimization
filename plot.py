import six
import sys
sys.modules['sklearn.externals.six'] = six
# import mlrose
import mlrose_hiive as mlrose

from functools import reduce
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt



labels = ['RHC', 'GA', 'SA', 'MIMIC']
colors = ['r', 'g', 'b', 'y']


for prefix in ["four_peaks", "k-color", "knapsack"]:

    rhc_run_stats = pd.read_csv('./' + prefix + '/RHC/rhc__RHC__run_stats_df.csv')
    ga_run_stats = pd.read_csv('./' + prefix + '/GA/ga__GA__run_stats_df.csv')
    sa_run_stats = pd.read_csv('./' + prefix + '/SA/sa__SA__run_stats_df.csv')
    mimic_run_stats = pd.read_csv('./' + prefix + '/MIMIC/mimic__MIMIC__run_stats_df.csv')


    stats = [rhc_run_stats, ga_run_stats, sa_run_stats, mimic_run_stats]


    iterations = rhc_run_stats['Iteration'].unique()


    for j, stat in enumerate(stats):
        avg_fitness = []
        clock_time = []
        evals = []
        for i in iterations:
            the_one = stat.loc[(stat['Iteration'] == i)].copy()
            data = the_one[['Iteration', 'Fitness', 'Time']].copy()
            avg_fitness.append(data['Fitness'].mean())
            clock_time.append(data['Time'].mean())

            if labels[j] == "RHC" or labels[j] == "SA":
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(1 + i + num_improves)
            else:
                data = the_one[['Iteration', 'Fitness', 'Time', 'Population Size']].copy()
                pop_size = data['Population Size'].mean()
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(pop_size + i * (pop_size + 1) + num_improves)

        plt.plot(iterations, avg_fitness, 'o-', color=colors[j], label=labels[j])

    plt.title(prefix + " Fitness vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel('Fitness')

    plt.legend(loc='best')
    plt.show()




    for j, stat in enumerate(stats):
        avg_fitness = []
        clock_time = []
        evals = []
        for i in iterations:
            the_one = stat.loc[(stat['Iteration'] == i)].copy()
            data = the_one[['Iteration', 'Fitness', 'Time']].copy()
            avg_fitness.append(data['Fitness'].mean())
            clock_time.append(data['Time'].mean())

            if labels[j] == "RHC" or labels[j] == "SA":
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(1 + i + num_improves)
            else:
                data = the_one[['Iteration', 'Fitness', 'Time', 'Population Size']].copy()
                pop_size = data['Population Size'].mean()
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(pop_size + i * (pop_size + 1) + num_improves)

        plt.plot(evals, avg_fitness, 'o-', color=colors[j], label=labels[j])

    plt.title(prefix + " Fitness vs Evals")
    plt.xlabel("Evals")
    plt.xlim(left=0)
    plt.xlim(right=2000)
    plt.ylabel('Fitness')

    plt.legend(loc='best')
    plt.show()

    for j, stat in enumerate(stats):
        avg_fitness = []
        clock_time = []
        evals = []
        for i in iterations:
            the_one = stat.loc[(stat['Iteration'] == i)].copy()
            data = the_one[['Iteration', 'Fitness', 'Time']].copy()
            avg_fitness.append(data['Fitness'].mean())
            clock_time.append(data['Time'].mean())

            if labels[j] == "RHC" or labels[j] == "SA":
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(1 + i + num_improves)
            else:
                data = the_one[['Iteration', 'Fitness', 'Time', 'Population Size']].copy()
                pop_size = data['Population Size'].mean()
                num_improves = 0
                prev_item = 0
                for item in data['Fitness']:
                    if item > prev_item:
                        num_improves += 1
                    prev_item = item
                evals.append(pop_size + i * (pop_size + 1) + num_improves)

        plt.plot(clock_time, avg_fitness, 'o-', color=colors[j], label=labels[j])

    plt.title(prefix + " Fitness vs Time")
    plt.xlabel("Time")
    plt.ylabel('Fitness')
    plt.xscale('log')

    plt.legend(loc='best')
    plt.show()



