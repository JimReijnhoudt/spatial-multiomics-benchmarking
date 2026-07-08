#####

# imports
import time
import pandas as pd
import os
import subprocess
import json

# results lists
results_list = []
results_metrics_list = []

# Setting default parameter values
# These parameter values are used in the tutorial of SpatialMETA and are assumed as the defaults
n_latent = 10
hidden_stacks = [128]
lr = 1e-3
max_epoch = 64
n_top_genes = 2000
n_top_metabolites = 800
kl_weight = 1
mmd_weight = 1

def run_spatialmeta():
    # print current parameter values
    print(f"n_latent: {n_latent}")
    print(f"hidden_stacks: {hidden_stacks}")
    print(f"lr: {lr}")
    print(f"max_epoch: {max_epoch}")
    print(f"n_top_genes: {n_top_genes}")
    print(f"n_top_metabolites: {n_top_metabolites}")
    print(f"kl_weight: {kl_weight}")
    print(f"mmd_weight: {mmd_weight}")
    print(f"starting run no. {iteration}")

    # create directory to save results
    iteration_dir = f"benchmarks/results/automated/plots/testdir/iteration_{iteration}"
    os.makedirs(iteration_dir, exist_ok=True)
    
    # running spatialmeta using a seperate python file
    subprocess.run(["python",
        "subprocess_vertical_spatialmeta.py",
        "-i", str(iteration_dir),
        "-l", str(n_latent),
        "-s", *map(str, hidden_stacks),
        "-r", str(lr),
        "-m", str(max_epoch),
        "-g", str(n_top_genes),
        "-b", str(n_top_metabolites),
        "-k", str(kl_weight),
        "-d", str(mmd_weight)])

    # opening the json files and saving the results as variables
    with open("results_run.json") as f:
        results_run = json.load(f)

    with open("results_metrics_run.json") as f:
        results_metrics_run = json.load(f)

    # adding parameter n_latent to the dictionary
    results_run["n_latent"] = n_latent
    results_metrics_run["n_latent"] = n_latent
    # adding parameter hidden_stacks to the dictionary
    results_run["hidden_stacks"] = hidden_stacks
    results_metrics_run["hidden_stacks"] = hidden_stacks
    # adding parameter lr to the dictionary
    results_run["lr"] = lr
    results_metrics_run["lr"] = lr
    # adding parameter max_epoch to the dictionary
    results_run["max_epoch"] = max_epoch
    results_metrics_run["max_epoch"] = max_epoch
    # adding parameter n_top_genes to the dictionary
    results_run["n_top_genes"] = n_top_genes
    results_metrics_run["n_top_genes"] = n_top_genes
    # adding parameter n_top_metabolites to the dictionary
    results_run["n_top_metabolites"] = n_top_metabolites
    results_metrics_run["n_top_metabolites"] = n_top_metabolites
    # adding parameter kl_weight to the dictionary
    results_run["kl_weight"] = kl_weight
    results_metrics_run["kl_weight"] = kl_weight
    # adding parameter mmd_weight to the dictionary
    results_run["mmd_weight"] = mmd_weight
    results_metrics_run["mmd_weight"] = mmd_weight
    # adding iteration_number to the dictionary
    results_run["iteration_number"] = iteration + 1
    results_metrics_run["iteration_number"] = iteration + 1
    # Adding the results dictionaries to the results lists

    # adding all results to the results list variables
    results_list.append(results_run)
    results_metrics_list.append(results_metrics_run)

    print(f"completed run no. {iteration}")


# running a parameter
for n_latent in [10, 20]:
    for iteration in range(2):
        run_spatialmeta()
    # saving results after each set of parameters
    results_df = pd.DataFrame(results_list)
    results_metrics_df = pd.DataFrame(results_metrics_list)
    results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
    results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)
# Reset default parameter values
n_latent = 10
hidden_stacks = [128]
lr = 1e-3
max_epoch = 64
n_top_genes = 2000
n_top_metabolites = 800
kl_weight = 1
mmd_weight = 1

# saving final results
results_df = pd.DataFrame(results_list)
results_metrics_df = pd.DataFrame(results_metrics_list)
results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)

# printing final results
print(results_metrics_df)
print("succes")