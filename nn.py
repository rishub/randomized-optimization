import numpy as np
import time
import six
import sys
sys.modules['sklearn.externals.six'] = six
# import mlrose
from mlrose_hiive import GeomDecay, ArithDecay, ExpDecay
from mlrose_hiive import NeuralNetwork


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('diabetes_2.csv') # https://www.openml.org/d/37
numeric = ["preg","plas","pres","skin","insu","mass","pedi","age"]
pos_label = "tested_positive"
df_num = df[numeric]
normalized_df=(df_num-df_num.min())/(df_num.max()-df_num.min()) # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
df = df.drop(numeric, axis=1)
df = pd.concat([df, normalized_df], axis=1)


# get train and test

x = df.drop("class", axis=1)
y = df["class"]

print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas



def plot():
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


def plot2():
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

for item in [25, 75]:
    print("RHC...")
    nn = NeuralNetwork(hidden_nodes=[30], 
                        activation='sigmoid',
                        algorithm='random_hill_climb',
                        max_iters=1024,
                        learning_rate=0.05,
                        early_stopping=True,
                        max_attempts=100,
                        restarts=item,
                        random_state=2,
                        curve=True)

    time.sleep(99999999)
    train_pred = nn.predict(x_train)
    test_pred = nn.predict(x_test)
    train = accuracy_score(y_true=y_train, y_pred=train_pred)
    test = accuracy_score(y_true=y_test, y_pred=test_pred)

    print("Test accuracy".format(item, test))
    print("Train accuracy".format(item, train))

for item in [GeomDecay, ArithDecay, ExpDecay]:
    print("SA")
    nn = mlrose.NeuralNetwork(hidden_nodes=[30], 
                                    activation='sigmoid',
                                   algorithm='simulated_annealing',
                                   schedule=decay(),
                                   max_iters=1024,
                                   learning_rate=0.05,
                                   early_stopping=True,
                                   max_attempts=100,
                                   random_state=2,
                                   curve=True)
    train = accuracy_score(y_true=y_train, y_pred=train_pred)
    test = accuracy_score(y_true=y_test, y_pred=test_pred)

    print("Test accuracy".format(item, test))
    print("Train accuracy".format(item, train))


for p in [150, 300]:
    for mut in [0.2, 0.4]:
        mlrose.NeuralNetwork(hidden_nodes=[30], 
                            activation='sigmoid',
                           algorithm='genetic_alg',
                           pop_size=pop, 
                           mutation_prob=mut,
                           max_iters=1024,
                           learning_rate=0.05,
                           early_stopping=True,
                           max_attempts=100,
                           random_state=2,
                           curve=True)
        train = accuracy_score(y_true=y_train, y_pred=train_pred)
        test = accuracy_score(y_true=y_test, y_pred=test_pred)

        print("Test accuracy".format(item, test))
        print("Train accuracy".format(item, train))


rhc_neural = mlrose.NeuralNetwork(hidden_nodes=[30], 
                                    activation='sigmoid',
                                    algorithm='random_hill_climb',
                                    max_iters=1024,
                                    learning_rate=0.05,
                                    early_stopping=True,
                                    max_attempts=100,
                                    random_state=2,
                                    curve=True)
sa_neural = mlrose.NeuralNetwork(hidden_nodes=[30], 
                                    activation='sigmoid',
                                   algorithm='simulated_annealing',
                                   schedule=GeomDecay(),
                                   max_iters=1024,
                                   learning_rate=0.05,
                                   early_stopping=True,
                                   max_attempts=100,
                                   random_state=2,
                                   curve=True)
ga_neural = mlrose.NeuralNetwork(hidden_nodes=[30], 
                                    activation='sigmoid',
                                   algorithm='genetic_alg',
                                   pop_size=300, 
                                   mutation_prob=0.2,
                                   max_iters=1024,
                                   learning_rate=0.05,
                                   early_stopping=True,
                                   max_attempts=100,
                                   random_state=2,
                                   curve=True)


for item in [rhc_neural, sa_neural, ga_neural]:
    t = time.time()
    item.fit(x_train, y_train)

    test_pred = item.predict(x_test)
    train_pred = item.predict(x_train)
    
    train = accuracy_score(y_true=y_train, y_pred=train_pred)
    test = accuracy_score(y_true=y_test, y_pred=test_pred)

    print("Test accuracy".format(item, test))
    print("Train accuracy".format(item, train))
    print("Time needed".format(time.time()-t))