#####

import time
import pandas as pd
import os
import subprocess

results_list = []
results_metrics_list = []

for n_latent in [5, 10]:
    for hidden_stacks in [128]:

        for iteration in range(2):

            print(f"n_latent: {n_latent}")
            print(f"hidden_stacks: {hidden_stacks}")

            print(f"starting run no. {iteration}")

            iteration_dir = f"benchmarks/results/automated/plots/latent{n_latent}_stacks{hidden_stacks}/iteration_{iteration}"
            os.makedirs(iteration_dir, exist_ok=True)
            
            start = time.perf_counter()

            # results_run, results_metrics_run = run_vertical_spatialmeta(iteration_dir, n_latent, hidden_stacks)
            
            subprocess.run(["python",
                "subprocess_vertical_spatialmeta.py",
                "-i", str(iteration_dir),
                "-l", str(n_latent),
                "-s", str(hidden_stacks)])

            runtime = time.perf_counter() - start

            # results_run["runtime"] = runtime
            # results_metrics_run["runtime"] = runtime

            # results_run["n_latent"] = n_latent
            # results_metrics_run["n_latent"] = n_latent
            
            # results_run["hidden_stacks"] = hidden_stacks
            # results_metrics_run["hidden_stacks"] = hidden_stacks

            # results_list.append(results_run)
            # results_metrics_list.append(results_metrics_run)

            print(f"completed run no. {iteration}")

# results_df = pd.DataFrame(results_metrics_list)
# results_metrics_df = pd.DataFrame(results_metrics_list)

# results_list.to_csv("benchmarks/metrics/automated/results_list.csv", index=False)

# results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
# results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)

# print(results_metrics_df)
print("succes")