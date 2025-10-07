from experiment3 import simulated_annealing
import numpy as np


if __name__ == "__main__":
    MAT = np.loadtxt(f"test-data/extra_credit.txt")
    sa_path, sa_cost, _ = simulated_annealing(MAT, T0=500, alpha=0.995, max_iters=10000)
    print(sa_path)
