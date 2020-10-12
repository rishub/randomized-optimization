import six
import sys
sys.modules['sklearn.externals.six'] = six
# import mlrose
import mlrose_hiive as mlrose

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


OUTPUT_DIRECTORY = "/Users/rishubkumar/Desktop/GaTech/7641/Assignment2/k-color"


edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
fitness_fn = mlrose.MaxKColor(edges)
problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness_fn, maximize=False, max_val=2)



iteration_list = 2 ** np.arange(10)

rhc = mlrose.RHCRunner(problem=problem,
                    experiment_name="RHC",
                    output_directory=OUTPUT_DIRECTORY,
                    seed=None,
                    iteration_list=iteration_list,
                    max_attempts=5000,
                    restart_list=[25, 75],
                    generate_curves=True,
                  )
rhc_run_stats, rhc_run_curves = rhc.run()


sa = mlrose.SARunner(problem=problem,
                    experiment_name="SA",
                    output_directory=OUTPUT_DIRECTORY,
                    seed=None,
                    iteration_list=iteration_list,
                    max_attempts=5000,
                    temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                    generate_curves=True,
                    decay_list=[mlrose.ExpDecay]
                    )
sa_run_stats, sa_run_curves = sa.run()


ga = mlrose.GARunner(problem=problem,
                      experiment_name="GA",
                      output_directory=OUTPUT_DIRECTORY,
                      seed=None,
                      iteration_list=iteration_list,
                      max_attempts=1000,
                      population_sizes=[150, 200, 300],
                      mutation_rates=[0.4, 0.5, 0.6],
                      generate_curves=True
                    )
ga_run_stats, ga_run_curves = ga.run()


mimic = mlrose.MIMICRunner(problem=problem,
                          experiment_name="MIMIC",
                          output_directory=OUTPUT_DIRECTORY,
                          seed=None,
                          iteration_list=iteration_list,
                          max_attempts=500,
                          keep_percent_list=[0.25, 0.5, 0.75],
                          population_sizes=[200],
                          generate_curves=True,
                          use_fast_mimic=True
                     )
mimic_run_stats, mimic_run_curves = mimic.run()