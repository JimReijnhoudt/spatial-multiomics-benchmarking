import subprocess

iteration_dir = "benchmarks/results/automated/plots/subprocesstest"
n_latent = 10
hidden_stacks = 128

subprocess.run(["python",
                "subprocess_vertical_spatialmeta.py",
                "-i", str(iteration_dir),
                "-l", str(n_latent),
                "-s", str(hidden_stacks)])

print("subprocess tester script succes")