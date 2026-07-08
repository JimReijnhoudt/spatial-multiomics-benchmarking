#####

import time
import pandas as pd
import os
import subprocess
import json

results_list = []
results_metrics_list = []

n_latent = 10
hidden_stacks = 128
lr = 1e-3
max_epoch = 64

for n_latent in [10]:
    for hidden_stacks in [128]:
        for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
            for max_epoch in [35, 64, 150]:

                for iteration in range(20):

                    print(f"n_latent: {n_latent}")
                    print(f"hidden_stacks: {hidden_stacks}")

                    print(f"lr: {lr}")
                    print(f"max_epoch: {max_epoch}")

                    print(f"starting run no. {iteration}")

                    iteration_dir = f"benchmarks/results/automated/plots/lr{str(lr)}_maxepoch{max_epoch}/iteration_{iteration}"
                    os.makedirs(iteration_dir, exist_ok=True)
                    
                    #start = time.perf_counter()
                    
                    subprocess.run(["python",
                        "subprocess_vertical_spatialmeta.py",
                        "-i", str(iteration_dir),
                        "-l", str(n_latent),
                        "-s", str(hidden_stacks),
                        "-r", str(lr),
                        "-m", str(max_epoch)])

                    #runtime = time.perf_counter() - start

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

                    # adding parameter n_latent to the dictionary
                    results_run["lr"] = lr
                    results_metrics_run["lr"] = lr

                    # adding parameter hidden_stacks to the dictionary
                    results_run["max_epoch"] = max_epoch
                    results_metrics_run["max_epoch"] = max_epoch

                    # adding iteration_number to the dictionary
                    results_run["iteration_number"] = iteration + 1
                    results_metrics_run["iteration_number"] = iteration + 1

                    # Adding the results dictionaries to the results lists
                    results_list.append(results_run)
                    results_metrics_list.append(results_metrics_run)

                    print(f"completed run no. {iteration}")

                results_df = pd.DataFrame(results_list)
                results_metrics_df = pd.DataFrame(results_metrics_list)

                results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
                results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)


results_df = pd.DataFrame(results_list)
results_metrics_df = pd.DataFrame(results_metrics_list)

# results_list.to_csv("benchmarks/results/automated/results.csv", index=False)

results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)

# print(results_df)
print(results_metrics_df)
print("succes")