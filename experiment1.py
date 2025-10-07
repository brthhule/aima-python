import time
import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

####################################### Algorithms
def calculate_cost(graph, tour):
    cost = 0
    for i in range(len(tour) - 1):
        cost += graph[tour[i]][tour[i + 1]]
    return cost

def nearest_neighbor(graph, start=0):
    n = len(graph)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    total_cost = 0
    current = start

    for _ in range(n - 1):
        nearest, nearest_dist = None, float("inf")
        for j in range(n):
            if not visited[j] and graph[current][j] < nearest_dist:
                nearest, nearest_dist = j, graph[current][j]
        tour.append(nearest)
        visited[nearest] = True
        total_cost += nearest_dist
        current = nearest

    total_cost += graph[current][start]
    tour.append(start)
    return tour, total_cost

def two_opt(graph, tour):
    best = tour
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                new_tour = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_cost(graph, new_tour) < calculate_cost(graph, best):
                    best = new_tour
                    improved = True
                    break
            if improved:
                break
    return best, calculate_cost(graph, best)

def nearest_neighbor_randomized(graph, k=2, start=0):
    """Choose next city randomly among k nearest unvisited ones."""
    n = len(graph)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        neighbors = [(graph[current][j], j) for j in range(n) if not visited[j]]
        neighbors.sort(key=lambda x: x[0])
        choices = neighbors[:min(k, len(neighbors))]
        _, next_city = random.choice(choices)
        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    tour.append(start)
    return tour, calculate_cost(graph, tour)

def rrnn_with_2opt(graph, k=2, num_repeats=10, start=0):
    best_tour, best_cost = None, float("inf")
    for rep in range(num_repeats):
        tour, cost = nearest_neighbor_randomized(graph, k, start)
        tour, cost = two_opt(graph, tour)
        if cost < best_cost:
            best_tour, best_cost = tour, cost
        # print(f"Repeat {rep+1}: Tour {tour}, Cost {cost:.2f}")
    return best_tour, best_cost

####################################### Experiment Helpers
def time_function(fn, *args, cpu_time_repeat_threshold=0, max_repeats_if_cpu_zero=100):
    t0 = time.time_ns()
    c0 = time.process_time_ns()
    result = fn(*args)
    c1 = time.process_time_ns()
    t1 = time.time_ns()
    wall = t1 - t0
    cpu = c1 - c0

    if cpu == 0 and max_repeats_if_cpu_zero > 1:
        runs = 0
        crepeat_start = time.process_time_ns()
        wrepeat_start = time.time_ns()
        while runs < max_repeats_if_cpu_zero:
            fn(*args)
            runs += 1
        crepeat_end = time.process_time_ns()
        wrepeat_end = time.time_ns()
        cpu = (crepeat_end - crepeat_start) // runs
        wall = (wrepeat_end - wrepeat_start) // runs
    return wall, cpu, result

def hyperparameter_sweep_k(matrices_sizes):
    medians_per_k = {}
    sizes = sorted(matrices_sizes.keys())
    for k in [1,2,3,4,5,7,10]:
        medians_per_size = []
        for size in sizes:
            scores = []
            for mat in matrices_sizes[size]:
                _, cost = rrnn_with_2opt(mat, k=k, num_repeats=5)
                scores.append(cost)
            medians_per_size.append(median(scores))
        medians_per_k[k] = medians_per_size
    return sizes, medians_per_k

def hyperparameter_sweep_num_repeats(matrices_sizes):
    medians_per_repeats = {}
    sizes = sorted(matrices_sizes.keys())
    for r in [1,2,5,10,20]:
        medians_per_size = []
        for size in sizes:
            scores = []
            for mat in matrices_sizes[size]:
                _, cost = rrnn_with_2opt(mat, k=3, num_repeats=r)
                scores.append(cost)
            medians_per_size.append(median(scores))
        medians_per_repeats[r] = medians_per_size
    return sizes, medians_per_repeats

def compare_algorithms(matrices_sizes, rrnn_k, rrnn_repeats):
    algorithms = ['NN', 'NN2', 'RRNN2']
    stats = {algo: {'sizes': [], 'median_wall_ns': [], 'median_cpu_ns': [], 'median_score': []} for algo in algorithms}

    for size, matrices in sorted(matrices_sizes.items()):
        per_algo_wall = {algo: [] for algo in algorithms}
        per_algo_cpu = {algo: [] for algo in algorithms}
        per_algo_score = {algo: [] for algo in algorithms}

        for mat in matrices:
            wall, cpu, res = time_function(nearest_neighbor, mat, 0)
            tour, cost = res
            per_algo_wall['NN'].append(wall)
            per_algo_cpu['NN'].append(cpu)
            per_algo_score['NN'].append(cost)

            wall, cpu, res = time_function(nearest_neighbor, mat, 0)
            tour, _ = res
            wall2, cpu2, res2 = time_function(two_opt, mat, tour)
            per_algo_wall['NN2'].append(wall + wall2)
            per_algo_cpu['NN2'].append(cpu + cpu2)
            per_algo_score['NN2'].append(res2[1])

            wall, cpu, res = time_function(rrnn_with_2opt, mat, rrnn_k, rrnn_repeats)
            tour, cost = res
            per_algo_wall['RRNN2'].append(wall)
            per_algo_cpu['RRNN2'].append(cpu)
            per_algo_score['RRNN2'].append(cost)

        for algo in algorithms:
            stats[algo]['sizes'].append(size)
            stats[algo]['median_wall_ns'].append(median(per_algo_wall[algo]))
            stats[algo]['median_cpu_ns'].append(median(per_algo_cpu[algo]))
            stats[algo]['median_score'].append(median(per_algo_score[algo]))

    return stats

####################################### Helpers

def plot_hyperparam_k(sizes, medians_per_k):
    plt.figure(figsize=(10,6))
    for k, medians in sorted(medians_per_k.items()):
        plt.plot(sizes, medians, marker='o', label=f'k={k}')
    plt.xlabel("Number of Cities")
    plt.ylabel("Median Tour Cost")
    plt.title("RRNN: Median Tour Cost vs Problem Size for different k")
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment1-images/hyperparam_k.png")
    plt.close()
    print("Saved experiment1-images/hyperparam_k.png")

def plot_hyperparam_repeats(sizes, medians_per_repeats):
    plt.figure(figsize=(10,6))
    for r, medians in sorted(medians_per_repeats.items()):
        plt.plot(sizes, medians, marker='o', label=f'repeats={r}')
    plt.xlabel("Number of Cities")
    plt.ylabel("Median Tour Cost")
    plt.title("RRNN: Median Tour Cost vs Problem Size for different num_repeats")
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment1-images/hyperparam_repeats.png")
    plt.close()
    print("Saved experiment1-images/hyperparam_repeats.png")

def plot_comparison_separate(stats):
    sizes = stats['NN']['sizes']
    x = sizes
    output_prefix="experiment1-images/comparison"

    # Wall time
    plt.figure(figsize=(8,6))
    plt.plot(x, np.array(stats['NN']['median_wall_ns'])/1e9, marker='o', label='NN')
    plt.plot(x, np.array(stats['NN2']['median_wall_ns'])/1e9, marker='s', label='NN+2opt')
    plt.plot(x, np.array(stats['RRNN2']['median_wall_ns'])/1e9, marker='^', label='RRNN+2opt')
    plt.xlabel("Number of Cities")
    plt.ylabel("Median Wall Time (s)")
    plt.title("Median Wall Time vs Number of Cities")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_wall_time.png")
    plt.close()
    print(f"Saved: {output_prefix}_wall_time.png")

    # CPU time
    plt.figure(figsize=(8,6))
    plt.plot(x, np.array(stats['NN']['median_cpu_ns'])/1e9, marker='o', label='NN')
    plt.plot(x, np.array(stats['NN2']['median_cpu_ns'])/1e9, marker='s', label='NN+2opt')
    plt.plot(x, np.array(stats['RRNN2']['median_cpu_ns'])/1e9, marker='^', label='RRNN+2opt')
    plt.xlabel("Number of Cities")
    plt.ylabel("Median CPU Time (s)")
    plt.title("Median CPU Time vs Number of Cities")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cpu_time.png")
    plt.close()
    print(f"Saved: {output_prefix}_cpu_time.png")

    # Tour cost
    plt.figure(figsize=(8,6))
    plt.plot(x, stats['NN']['median_score'], marker='o', label='NN')
    plt.plot(x, stats['NN2']['median_score'], marker='s', label='NN+2opt')
    plt.plot(x, stats['RRNN2']['median_score'], marker='^', label='RRNN+2opt')
    plt.xlabel("Number of Cities")
    plt.ylabel("Median Tour Cost (distance)")
    plt.title("Median Tour Cost vs Number of Cities")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_tour_cost.png")
    plt.close()
    print(f"Saved: {output_prefix}_tour_cost.png")


####################################### Main Section

if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    print("Loading matrices...")
    matrices_sizes = {}
    for a in [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]:
        matrices = []
        for b in range(4):
            MAT = np.loadtxt(f"test-data/{a}_random_adj_mat_{b}.txt")
            matrices.append(MAT)
        matrices_sizes[a] = matrices

    print("\nHyperparameter tuning...")
    sizes_k, medians_per_k = hyperparameter_sweep_k(matrices_sizes)
    plot_hyperparam_k(sizes_k, medians_per_k)

    sizes_r, medians_per_repeats = hyperparameter_sweep_num_repeats(matrices_sizes)
    plot_hyperparam_repeats(sizes_r, medians_per_repeats)

    # Hyperparameters type
    best_k = min(medians_per_k, key=lambda kk: medians_per_k[kk])
    best_r = min(medians_per_repeats, key=lambda rr: medians_per_repeats[rr])

    print("\nComparing algs")
    stats = compare_algorithms(matrices_sizes, rrnn_k=best_k, rrnn_repeats=best_r)
    plot_comparison_separate(stats)

    print(f"Best median k: {best_k}")
    print(f"Best median num_repeats: {best_r}")
