#Copyright (C) 2022 Intel Corporation*<br>
#SPDX-License-Identifier: BSD-3-Clause*<br>
#See: https://spdx.org/licenses/*
# Interface for QUBO problems

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, solve
import os
import numpy as np
import networkx as ntx
#If Loihi 2 hardware is available, we can take advantage of the large speed and energy efficiency of this chip to solve QUBOs. To access the chip, we must configure the following environment variables:

from lava.utils.system import Loihi2
Loihi2.preferred_partition = "oheogulch"
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    # Enable SLURM, the workload manager used to distribute Loihi2 resources to users
    os.environ['SLURM'] = '1'
    os.environ["PARTITION"] = "oheogulch"

example_workloads = {
    1: {'n_vert': 500, 'p_edge': 0.9, 'seed_graph': 5530, 'w_diag': 1, 'w_off': 3, 'cost_optimal': -5.0},
    2: {'n_vert': 45, 'p_edge': 0.5, 'seed_graph': 7865, 'w_diag': 1, 'w_off': 4, 'cost_optimal': -7.0, },
    3: {'n_vert': 45, 'p_edge': 0.7, 'seed_graph': 7079, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -33.0},
    }

workload = example_workloads[1]

# Import utility functions to create and analyze MIS workloads
from lava.lib.optimization.utils.generators.mis import MISProblem

# Create an undirected graph with 700 vertices and a 
# probability of 70% that any two vertices are randomly connected
mis = MISProblem(num_vertices=workload['n_vert'], connection_prob=workload['p_edge'], seed=workload['seed_graph'])

# Translate the MIS problem for this graph into a QUBO matrix
w_mult = 1 # CAN BE ADJUSTED
q = mis.get_qubo_matrix(w_diag=w_mult*workload['w_diag'], w_off=w_mult*workload['w_off'])

# Create the qubo problem
qubo_problem = QUBO(q)

# Find the optimal solution to the MIS problem
solution_opt = mis.find_maximum_independent_set()

# Calculate the QUBO cost of the optimal solution
cost_opt = qubo_problem.evaluate_cost(solution=solution_opt)

solver = OptimizationSolver(qubo_problem)

print(cost_opt)

#Solve the qubo problem with a set of hyperparameters

solver = OptimizationSolver(qubo_problem)

def stop(best_cost, best_to_sol):
     return best_cost <= cost_opt

#Stochastic search for a set of optimal hyperparameters with SolverTuner

from lava.lib.optimization.utils.solver_tuner import SolverTuner

search_space = {
    #"steps_to_fire": (9, 11),
    "noise_amplitude": ( 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "noise_precision": (2, 3, 4, 5),
    "step_size": (100, 105, 110, 111, 112, 113, 114,  115, 116, 117, 118, 119, 120, 125, 130, )
}
search_space, params_names = SolverTuner.generate_grid(search_space)
solver_tuner = SolverTuner(search_space=search_space, params_names= params_names, shuffle= True )

params = {"timeout": 100000,
          "target_cost": int(cost_opt),
          "backend": "CPU"}
    #print(params)
def fitness(cost, step_to_sol):
        return -cost 

fitness_target = -cost_opt

hyperparams, success = solver_tuner.tune(
                solver=solver,
                solver_params=params,
                fitness_fn=fitness,
                fitness_target=fitness_target
            )
print(f"{hyperparams, success}")

result = solver_tuner.results
import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
df.to_csv("grid_search_results")
sns.heatmap(df.pivot_table(index=["noise_amplitude", "step_size"], values="cost", aggfunc="max"))
import matplotlib.pyplot as plt
plt.savefig('grid_search_results.png')

print(result)
# Find the optimal solution to the MIS problem
solution_opt = mis.find_maximum_independent_set()

# Calculate the QUBO cost of the optimal solution
cost_opt = qubo_problem.evaluate_cost(solution=solution_opt)
