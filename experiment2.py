import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import heapq
import time

#######################################
# Timing helper
#######################################
def time_function(func, *args, **kwargs):
    t0 = time.time_ns()
    cpu0 = time.process_time_ns()
    result = func(*args, **kwargs)
    t1 = time.time_ns()
    cpu1 = time.process_time_ns()
    return t1-t0, cpu1-cpu0, result

#######################################
# Nearest Neighbor
#######################################
def nearest_neighbor(graph):
    n = graph.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(1, n):
        last = tour[-1]
        next_city = min((i for i in range(n) if not visited[i]), key=lambda i: graph[last][i])
        tour.append(next_city)
        visited[next_city] = True

    tour.append(0)
    cost = sum(graph[tour[i]][tour[i+1]] for i in range(n))
    return tour, cost

#######################################
# 2-Opt
#######################################
def two_opt(graph, tour):
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                if j-i == 1: 
                    continue
                delta = (graph[tour[i-1]][tour[j-1]] + graph[tour[i]][tour[j]] -
                         graph[tour[i-1]][tour[i]] - graph[tour[j-1]][tour[j]])
                if delta < 0:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
    cost = sum(graph[tour[i]][tour[i+1]] for i in range(n))
    return tour, cost

#######################################
# RRNN + 2-Opt
#######################################
def rrnn_with_2opt(graph, k=3, num_repeats=5):
    n = graph.shape[0]
    best_tour = None
    best_cost = float('inf')

    for _ in range(num_repeats):
        visited = [False]*n
        tour = [0]
        visited[0] = True

        for _ in range(1, n):
            last = tour[-1]
            candidates = [(i, graph[last][i]) for i in range(n) if not visited[i]]
            candidates.sort(key=lambda x: x[1])
            next_city = random.choice(candidates[:k])[0]
            tour.append(next_city)
            visited[next_city] = True

        tour.append(0)
        tour, cost = two_opt(graph, tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    return best_tour, best_cost

#######################################
# MST Heuristic & A* TSP
#######################################
def prim_mst_cost(submat):
    n = submat.shape[0]
    if n <= 1: return 0.0
    in_mst = [False]*n
    min_cost = [float('inf')]*n
    min_cost[0] = 0
    total = 0
    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: min_cost[i])
        total += min_cost[u]
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v]:
                min_cost[v] = min(min_cost[v], submat[u][v])
    return total

def mst_heuristic(graph, current_city, visited_mask, n, start=0):
    unvisited = [i for i in range(n) if not ((visited_mask >> i) & 1)]
    if not unvisited: return graph[current_city][start]
    idx = np.ix_(unvisited, unvisited)
    submat = graph[idx]
    mst_cost = prim_mst_cost(submat)
    min_from_current = min(graph[current_city][i] for i in unvisited)
    min_to_start = min(graph[i][start] for i in unvisited)
    return mst_cost + min_from_current + min_to_start

def astar_tsp(graph, start=0, time_limit_seconds=None):
    n = graph.shape[0]
    ALL_VISITED = (1 << n) - 1
    pq = []
    start_mask = 1 << start
    heapq.heappush(pq, (mst_heuristic(graph, start, start_mask, n), 0, start, start_mask, (start,)))
    best_g = {}
    best_solution = None
    best_cost = float('inf')
    nodes_expanded = 0
    t0 = time.time()
    
    while pq:
        if time_limit_seconds is not None and (time.time() - t0) > time_limit_seconds:
            break
        f, g, current, mask, path = heapq.heappop(pq)
        nodes_expanded += 1
        key = (current, mask)
        if key in best_g and g > best_g[key]:
            continue
        if mask == ALL_VISITED:
            total_cost = g + graph[current][start]
            full_path = path + (start,)
            if total_cost < best_cost:
                best_solution = list(full_path)
                best_cost = total_cost
            return best_solution, best_cost, nodes_expanded
        for next_city in range(n):
            if (mask >> next_city) & 1: continue
            g2 = g + graph[current][next_city]
            mask2 = mask | (1 << next_city)
            path2 = path + (next_city,)
            f2 = g2 + mst_heuristic(graph, next_city, mask2, n)
            if g2 < best_g.get((next_city, mask2), float('inf')):
                best_g[(next_city, mask2)] = g2
                heapq.heappush(pq, (f2, g2, next_city, mask2, path2))
    return best_solution, best_cost, nodes_expanded

#######################################
# Hyperparameter Sweeps
#######################################
def hyperparameter_sweep_k(matrices):
    medians_per_k = {}
    for k in [1,2,3,4,5,7,10]:
        scores = []
        for mat in matrices:
            _, cost = rrnn_with_2opt(mat, k=k, num_repeats=5)
            scores.append(cost)
        medians_per_k[k] = median(scores)
    return medians_per_k

def hyperparameter_sweep_num_repeats(matrices, num_repeats_values, fixed_k):
    medians_per_repeats = {}
    for r in num_repeats_values:
        scores = []
        for mat in matrices:
            _, cost = rrnn_with_2opt(mat, k=fixed_k, num_repeats=r)
            scores.append(cost)
        medians_per_repeats[r] = median(scores)
    return medians_per_repeats

#######################################
# Compare algorithms including A*
#######################################
def compare_algorithms_with_astar(matrices_sizes, rrnn_k, rrnn_repeats):
    algorithms = ['NN','NN2','RRNN2','A*']
    stats = {algo:{'sizes':[], 'median_wall_ns':[], 'median_cpu_ns':[], 'median_score':[]} for algo in algorithms}
    stats['A*']['median_nodes'] = []

    for size, matrices in matrices_sizes.items():
        per_algo_wall = {algo: [] for algo in algorithms}
        per_algo_cpu = {algo: [] for algo in algorithms}
        per_algo_score = {algo: [] for algo in algorithms}
        nodes_per_matrix = []

        for mat in matrices:
            wall, cpu, res = time_function(nearest_neighbor, mat)
            tour, cost = res
            per_algo_wall['NN'].append(wall)
            per_algo_cpu['NN'].append(cpu)
            per_algo_score['NN'].append(cost)

            wall, cpu, res = time_function(nearest_neighbor, mat)
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

            if size <= 15:
                wall, cpu, res = time_function(astar_tsp, mat)
                tour, cost, nodes_expanded = res
                per_algo_wall['A*'].append(wall)
                per_algo_cpu['A*'].append(cpu)
                per_algo_score['A*'].append(cost)
                nodes_per_matrix.append(nodes_expanded)

        for algo in algorithms:
            if per_algo_score[algo]:
                stats[algo]['sizes'].append(size)
                stats[algo]['median_wall_ns'].append(median(per_algo_wall[algo]))
                stats[algo]['median_cpu_ns'].append(median(per_algo_cpu[algo]))
                stats[algo]['median_score'].append(median(per_algo_score[algo]))
        if nodes_per_matrix:
            stats['A*']['median_nodes'].append(median(nodes_per_matrix))
    return stats

#######################################
# Plot normalized comparison
#######################################
def plot_comparison_normalized(stats, output_file="normalized_comparison.png"):
    # Only consider sizes where A* ran
    x = stats['A*']['sizes']

    plt.figure(figsize=(16,12))

    # Wall time / A*
    plt.subplot(3,1,1)
    for algo in ['NN','NN2','RRNN2']:
        wall_ns = [stats[algo]['median_wall_ns'][stats[algo]['sizes'].index(size)] for size in x]
        plt.plot(x, np.array(wall_ns)/np.array(stats['A*']['median_wall_ns']),
                 marker='o', label=f"{algo}/A* Wall")

    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        line_nodes, = ax2.plot(x, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")

    plt.ylabel("Wall Time / A*")
    plt.xlabel("Number of Cities")
    plt.xticks(x)
    plt.title("Algorithm Wall Time Normalized by A* with Nodes Expanded")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2)
    plt.grid(True)

    # CPU time / A*
    plt.subplot(3,1,2)
    for algo in ['NN','NN2','RRNN2']:
        cpu_ns = [stats[algo]['median_cpu_ns'][stats[algo]['sizes'].index(size)] for size in x]
        plt.plot(x, np.array(cpu_ns)/np.array(stats['A*']['median_cpu_ns']),
                 marker='o', label=f"{algo}/A* CPU")

    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        line_nodes, = ax2.plot(x, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")

    plt.ylabel("CPU Time / A*")
    plt.xlabel("Number of Cities")
    plt.xticks(x)
    plt.title("Algorithm CPU Time Normalized by A* with Nodes Expanded")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2)
    plt.grid(True)

    # Tour cost / A*
    plt.subplot(3,1,3)
    for algo in ['NN','NN2','RRNN2']:
        cost = [stats[algo]['median_score'][stats[algo]['sizes'].index(size)] for size in x]
        plt.plot(x, np.array(cost)/np.array(stats['A*']['median_score']),
                 marker='o', label=f"{algo}/A* Cost")

    plt.ylabel("Tour Cost / A*")
    plt.xlabel("Number of Cities")
    plt.xticks(x)
    plt.title("Algorithm Tour Cost Normalized by A*")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


#######################################
# Main
#######################################
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Load adjacency matrices (example: n=5,10,15)
    matrices_sizes = {}
    for size in [5,10,15,20]:
        matrices = []
        for b in range(10):
            MAT = np.loadtxt(f"test-data/{size}_random_adj_mat_{b}.txt")
            matrices.append(MAT)
        matrices_sizes[size] = matrices

    # Hyperparameter sweep
    medians_per_k = hyperparameter_sweep_k(matrices_sizes[15])
    best_k = min(medians_per_k, key=medians_per_k.get)
    medians_per_repeats = hyperparameter_sweep_num_repeats(matrices_sizes[15],[1,2,5,10,20,30], fixed_k=best_k)
    best_r = min(medians_per_repeats, key=medians_per_repeats.get)
    print(f"Best k: {best_k}, Best num_repeats: {best_r}")

    # Compare algorithms
    stats = compare_algorithms_with_astar(matrices_sizes, rrnn_k=best_k, rrnn_repeats=best_r)

    # Plot normalized comparison
    plot_comparison_normalized(stats)
