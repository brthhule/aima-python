import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import heapq
import time
from experiment1 import nearest_neighbor, two_opt, rrnn_with_2opt

def time_function(func, *args, **kwargs):
    t0 = time.time_ns()
    cpu0 = time.process_time_ns()
    result = func(*args, **kwargs)
    t1 = time.time_ns()
    cpu1 = time.process_time_ns()
    return t1-t0, cpu1-cpu0, result

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
            continue
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

####################################### Hyperparameters
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

def compare_algorithms_with_astar(matrices_sizes, rrnn_k, rrnn_repeats):
    algorithms = ['NN','NN2','RRNN2','A*']
    stats = {algo:{'sizes':[], 'median_wall_ns':[], 'median_cpu_ns':[], 'median_score':[]} for algo in algorithms}
    stats['A*']['median_nodes'] = []

    for size, matrices in matrices_sizes.items():
        print("Running size ", size)
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

def plot_comparison_normalized(stats, output_dir="experiment2-images"):
    sizes_with_astar = stats['A*']['sizes']
    if not sizes_with_astar:
        print("No sizes with A* results, skipping plot.")
        return

    # Wall time / A* (with twin axis and legend)
    plt.figure(figsize=(10, 6))
    for algo in ['NN', 'NN2', 'RRNN2']:
        wall_ns = []
        a_star_wall_ns = []
        for size in sizes_with_astar:
            if size in stats[algo]['sizes']:
                idx_algo = stats[algo]['sizes'].index(size)
                idx_astar = stats['A*']['sizes'].index(size)
                wall_ns.append(stats[algo]['median_wall_ns'][idx_algo])
                a_star_wall_ns.append(stats['A*']['median_wall_ns'][idx_astar])
        plt.plot(sizes_with_astar, np.array(wall_ns) / np.where(np.array(a_star_wall_ns) == 0, 1e-7, np.array(a_star_wall_ns)), 
                 marker='o', label=f"{algo}/A* Wall")
    
    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        ax2.plot(sizes_with_astar, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")

    plt.xlabel("Number of Cities")
    plt.ylabel("Wall Time / A*")
    plt.title("Algorithm Wall Time Normalized by A* with Nodes Expanded")

    # Combine legends from both axes
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_wall_time.png")
    plt.close()
    print(f"Saved: {output_dir}/normalized_wall_time.png")

    # CPU time / A* (no legend needed for twin axis here)
    plt.figure(figsize=(10, 6))
    for algo in ['NN', 'NN2', 'RRNN2']:
        cpu_ns = []
        a_star_cpu_ns = []
        for size in sizes_with_astar:
            if size in stats[algo]['sizes']:
                idx_algo = stats[algo]['sizes'].index(size)
                idx_astar = stats['A*']['sizes'].index(size)
                cpu_ns.append(stats[algo]['median_cpu_ns'][idx_algo])
                a_star_cpu_ns.append(stats['A*']['median_cpu_ns'][idx_astar])
        plt.plot(sizes_with_astar, np.array(cpu_ns) / np.where(np.array(a_star_cpu_ns) == 0, 1e-7, np.array(a_star_cpu_ns)),
                 marker='o', label=f"{algo}/A* CPU")
    

    print("Running normalized CPU time")
    print(stats)
    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        ax2.plot(sizes_with_astar, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")

    plt.xlabel("Number of Cities")
    plt.ylabel("CPU Time / A*")
    plt.title("Algorithm CPU Time Normalized by A* with Nodes Expanded")
    plt.grid(True)
    plt.tight_layout()
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.savefig(f"{output_dir}/normalized_cpu_time.png")
    plt.close()
    print(f"Saved: {output_dir}/normalized_cpu_time.png")

    plt.figure(figsize=(10, 6))
    for algo in ['NN', 'NN2', 'RRNN2']:
        cost = []
        a_star_cost = []
        for size in sizes_with_astar:
            if size in stats[algo]['sizes']:
                idx_algo = stats[algo]['sizes'].index(size)
                idx_astar = stats['A*']['sizes'].index(size)
                cost.append(stats[algo]['median_score'][idx_algo])
                a_star_cost.append(stats['A*']['median_score'][idx_astar])
        plt.plot(sizes_with_astar, np.array(cost) / np.where(np.array(a_star_cost) == 0, 1e-7, np.array(a_star_cost)),
                 marker='o', label=f"{algo}/A* Cost")
    
    plt.xlabel("Number of Cities")
    plt.ylabel("Tour Cost / A*")
    plt.title("Algorithm Tour Cost Normalized by A*")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_tour_cost.png")
    plt.close()
    print(f"Saved: {output_dir}/normalized_tour_cost.png")

def plot_astar_nodes(stats, output_dir="experiment2-images"):
    sizes_with_astar = stats['A*']['sizes']
    if 'median_nodes' not in stats['A*'] or not sizes_with_astar:
        print("No A* nodes data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_with_astar, stats['A*']['median_nodes'], marker='o', color='purple', label="A* Median Nodes")
    plt.xlabel("Number of Cities")
    plt.ylabel("Median Nodes Expanded by A*")
    plt.title("A* Nodes Expanded vs Number of Cities")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/astar_nodes_expanded.png")
    plt.close()
    print(f"Saved: {output_dir}/astar_nodes_expanded.png")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    matrices_sizes = {}
    for size in [5, 6, 7, 8, 9, 10, 15]:
        matrices = []
        for b in range(10):
            MAT = np.loadtxt(f"test-data/{size}_random_adj_mat_{b}.txt")
            matrices.append(MAT)
        matrices_sizes[size] = matrices

    medians_per_k = hyperparameter_sweep_k(matrices_sizes[15])
    best_k = min(medians_per_k, key=medians_per_k.get)
    medians_per_repeats = hyperparameter_sweep_num_repeats(matrices_sizes[15],[1,2,5,10,20,30], fixed_k=best_k)
    best_r = min(medians_per_repeats, key=medians_per_repeats.get)
    print(f"Best k: {best_k}, Best num_repeats: {best_r}")

    print("Comparing algorithms...")
    stats = compare_algorithms_with_astar(matrices_sizes, rrnn_k=best_k, rrnn_repeats=best_r)

    plot_astar_nodes(stats)

    plot_comparison_normalized(stats)
