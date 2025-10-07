#!/usr/bin/env python3
"""
experiment3_full.py

Runs Hill Climbing, Simulated Annealing, and Genetic Algorithm on TSP adjacency matrices
from test-data/*.txt. Generates plots for hyperparameters, convergence, and comparisons.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import os
import time
from experiment2 import astar_tsp

os.makedirs("experiment3-images", exist_ok=True)

def calculate_cost(graph, tour):
    return sum(graph[tour[i]][tour[i+1]] for i in range(len(tour)-1))

def time_function(func, *args, **kwargs):
    t0 = time.time_ns()
    cpu0 = time.process_time_ns()
    result = func(*args, **kwargs)
    t1 = time.time_ns()
    cpu1 = time.process_time_ns()
    return t1 - t0, cpu1 - cpu0, result

def hill_climbing_random_restarts(graph, num_restarts=10, inner_iters=200):
    print("Running hill_climbing_random_restarts")
    n = graph.shape[0]
    best_overall = None
    best_cost_overall = float('inf')
    overall_history = []
    for _ in range(num_restarts):
        perm = list(range(n))
        random.shuffle(perm)
        tour = perm + [perm[0]]
        best_local = tour[:]
        best_local_cost = calculate_cost(graph, tour)
        local_history = []
        for _ in range(inner_iters):
            i, j = random.sample(range(1, n), 2)
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_cost = calculate_cost(graph, new_tour)
            if new_cost < best_local_cost:
                best_local = new_tour[:]
                best_local_cost = new_cost
                tour = new_tour
            local_history.append(best_local_cost)
        if best_local_cost < best_cost_overall:
            best_cost_overall = best_local_cost
            best_overall = best_local[:]
        if not overall_history:
            overall_history = local_history[:]
        else:
            L = max(len(overall_history), len(local_history))
            while len(overall_history) < L:
                overall_history.append(best_cost_overall)
            while len(local_history) < L:
                local_history.append(local_history[-1])
            overall_history = [min(a,b) for a,b in zip(overall_history, local_history)]
    return best_overall, best_cost_overall, overall_history

def simulated_annealing(graph, T0=100.0, alpha=0.995, max_iters=200):
    print("Running simulated_annealing")
    n = graph.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    tour = perm + [perm[0]]
    cost = calculate_cost(graph, tour)
    best_tour = tour[:]
    best_cost = cost
    history = [best_cost]
    T = T0
    for _ in range(max_iters):
        i, j = random.sample(range(1, n), 2)
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_cost = calculate_cost(graph, new_tour)
        delta = new_cost - cost
        if delta < 0 or random.random() < np.exp(-delta/T):
            tour = new_tour
            cost = new_cost
            T *= alpha
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
        history.append(best_cost)
        if T < 1e-12:
            T = 1e-12
    return best_tour, best_cost, history

def ordered_crossover(p1, p2):
    n = len(p1)-1
    p1, p2 = p1[:-1], p2[:-1]
    a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n
    child[a:b] = p1[a:b]
    fill_pos = b % n
    i = b
    while -1 in child:
        gene = p2[i%n]
        if gene not in child:
            child[fill_pos] = gene
            fill_pos = (fill_pos+1)%n
        i += 1
    child.append(child[0])
    return child

def mutate_swap(tour, mutation_chance):
    n = len(tour)-1
    new = tour[:]
    if random.random() < mutation_chance:
        i, j = random.sample(range(1,n),2)
        new[i], new[j] = new[j], new[i]
    return new

def genetic_algorithm(graph, population_size=50, mutation_chance=0.1, num_generations=200):
    print("Running genetic_algorithm")
    n = graph.shape[0]
    population = []
    for _ in range(population_size):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm + [perm[0]])
    fitness = [calculate_cost(graph, t) for t in population]
    history = []
    for _ in range(num_generations):
        new_children = []
        while len(new_children) < population_size:
            def tournament():
                k = min(5, population_size)
                contestants = random.sample(range(population_size), k)
                best = min(contestants, key=lambda idx: fitness[idx])
                return population[best]
            p1, p2 = tournament(), tournament()
            child = ordered_crossover(p1,p2)
            child = mutate_swap(child, mutation_chance)
            new_children.append(child)
        combined = population + new_children
        combined_costs = [calculate_cost(graph,t) for t in combined]
        idxs = sorted(range(len(combined)), key=lambda i: combined_costs[i])[:population_size]
        population = [combined[i] for i in idxs]
        fitness = [combined_costs[i] for i in idxs]
        history.append(min(fitness))
    best_idx = int(np.argmin(fitness))
    return population[best_idx], fitness[best_idx], history



def hyperparameter_sweep_sa(matrices, T0_values=[10,50,100,200], alpha_values=[0.99,0.995,0.999], max_iters=500):
    print("Running hyperparameter_sweep_sa")
    results_T0 = {}
    results_alpha = {}
    default_alpha = alpha_values[len(alpha_values)//2]
    for T0 in T0_values:
        costs = []
        for mat in matrices:
            _, cost, _ = simulated_annealing(mat, T0=T0, alpha=default_alpha, max_iters=max_iters)
            costs.append(cost)
        results_T0[T0] = median(costs)
    default_T0 = T0_values[len(T0_values)//2]
    for alpha in alpha_values:
        costs = []
        for mat in matrices:
            _, cost, _ = simulated_annealing(mat, T0=default_T0, alpha=alpha, max_iters=max_iters)
            costs.append(cost)
        results_alpha[alpha] = median(costs)
    return results_T0, results_alpha

def hyperparameter_sweep_ga(matrices, pop_sizes=[20,50,100], mutation_chances=[0.01,0.05,0.1], num_generations=200):
    print("Running hyperparameter_sweep_ga")
    results_pop = {}
    results_mut = {}
    default_mut = mutation_chances[len(mutation_chances)//2]
    for ps in pop_sizes:
        costs = []
        for mat in matrices:
            _, cost, _ = genetic_algorithm(mat, population_size=ps, mutation_chance=default_mut, num_generations=num_generations)
            costs.append(cost)
        results_pop[ps] = median(costs)
    default_pop = pop_sizes[len(pop_sizes)//2]
    for mu in mutation_chances:
        costs = []
        for mat in matrices:
            _, cost, _ = genetic_algorithm(mat, population_size=default_pop, mutation_chance=mu, num_generations=num_generations)
            costs.append(cost)
        results_mut[mu] = median(costs)
    return results_pop, results_mut

def compare_part3_algorithms(matrices_sizes, hc_restarts, sa_params, ga_params, astar_func):
    print("Running compare_part3_algorithms")
    algorithms = ['HC','SA','GA','A*']
    stats = {algo:{'sizes':[], 'median_wall_ns':[], 'median_cpu_ns':[], 'median_score':[]} for algo in algorithms}
    stats['A*']['median_nodes'] = []

    for size, mats in sorted(matrices_sizes.items()):
        per_wall = {a:[] for a in algorithms}
        per_cpu = {a:[] for a in algorithms}
        per_score = {a:[] for a in algorithms}
        nodes_per_matrix = []

        for mat in mats:
            wall, cpu, res = time_function(hill_climbing_random_restarts, mat, hc_restarts, 500)
            tour, cost, _ = res
            per_wall['HC'].append(wall); per_cpu['HC'].append(cpu); per_score['HC'].append(cost)

            wall, cpu, res = time_function(simulated_annealing, mat, sa_params.get('T0',100.0), sa_params.get('alpha',0.995), sa_params.get('max_iters',500))
            tour, cost, _ = res
            per_wall['SA'].append(wall); per_cpu['SA'].append(cpu); per_score['SA'].append(cost)

            wall, cpu, res = time_function(genetic_algorithm, mat, ga_params.get('population_size',50), ga_params.get('mutation_chance',0.1), ga_params.get('num_generations',200))
            tour, cost, _ = res
            per_wall['GA'].append(wall); per_cpu['GA'].append(cpu); per_score['GA'].append(cost)

            wall, cpu, res = time_function(astar_func, mat)
            tour, cost, nodes = res
            per_wall['A*'].append(wall); per_cpu['A*'].append(cpu); per_score['A*'].append(cost)
            nodes_per_matrix.append(nodes)

        for algo in algorithms:
            stats[algo]['sizes'].append(size)
            stats[algo]['median_wall_ns'].append(median(per_wall[algo]))
            stats[algo]['median_cpu_ns'].append(median(per_cpu[algo]))
            stats[algo]['median_score'].append(median(per_score[algo]))
        if nodes_per_matrix:
            stats['A*']['median_nodes'].append(median(nodes_per_matrix))
    return stats


def plot_normalized_part3(stats, output_dir="experiment3-images"):
    print("Normalizing plots...")
    sizes = stats['A*']['sizes']
    if not sizes:
        print("No A* sizes found — cannot normalize.")
        return

    algos = ['HC','SA','GA']
    # Wall
    plt.figure(figsize=(10,6))
    for algo in algos:
        vals = []
        a_vals = []
        for s in sizes:
            if s in stats[algo]['sizes']:
                idx_a = stats[algo]['sizes'].index(s)
                idx_ast = stats['A*']['sizes'].index(s)
                vals.append(stats[algo]['median_wall_ns'][idx_a])
                a_vals.append(stats['A*']['median_wall_ns'][idx_ast])
        plt.plot(sizes, np.array(vals) / np.where(np.array(a_vals)==0,1e-9,np.array(a_vals)), marker='o', label=f"{algo}/A* Wall")
    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        ax2.plot(sizes, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")
    plt.xlabel("Number of Cities"); plt.ylabel("Wall Time / A*"); plt.title("Wall Time Normalized by A*")
    lines, labels = plt.gca().get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    plt.legend(lines + l2, labels + lab2, loc="upper left")
    plt.grid(True); plt.tight_layout(); plt.savefig(f"{output_dir}/part3_normalized_wall.png"); plt.close()

    # CPU
    plt.figure(figsize=(10,6))
    for algo in algos:
        vals = []; a_vals = []
        for s in sizes:
            if s in stats[algo]['sizes']:
                idx_a = stats[algo]['sizes'].index(s)
                idx_ast = stats['A*']['sizes'].index(s)
                vals.append(stats[algo]['median_cpu_ns'][idx_a])
                a_vals.append(stats['A*']['median_cpu_ns'][idx_ast])
        plt.plot(sizes, np.array(vals) / np.where(np.array(a_vals)==0,1e-9,np.array(a_vals)), marker='o', label=f"{algo}/A* CPU")
    ax2 = plt.gca().twinx()
    if 'median_nodes' in stats['A*']:
        ax2.plot(sizes, stats['A*']['median_nodes'], 'k--', label="A* Nodes")
        ax2.set_ylabel("Median Nodes Expanded by A*")
    plt.xlabel("Number of Cities"); plt.ylabel("CPU Time / A*"); plt.title("CPU Time Normalized by A*")
    lines, labels = plt.gca().get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    plt.legend(lines + l2, labels + lab2, loc="upper left")
    plt.grid(True); plt.tight_layout(); plt.savefig(f"{output_dir}/part3_normalized_cpu.png"); plt.close()


    plt.figure(figsize=(10,6))
    for algo in algos:
        vals = []; a_vals = []
        for s in sizes:
            if s in stats[algo]['sizes']:
                idx_a = stats[algo]['sizes'].index(s)
                idx_ast = stats['A*']['sizes'].index(s)
                vals.append(stats[algo]['median_score'][idx_a])
                a_vals.append(stats['A*']['median_score'][idx_ast])
        plt.plot(sizes, np.array(vals) / np.where(np.array(a_vals)==0,1e-9,np.array(a_vals)), marker='o', label=f"{algo}/A* Cost")
    plt.xlabel("Number of Cities"); plt.ylabel("Cost / A*"); plt.title("Cost Normalized by A*")
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout(); plt.savefig(f"{output_dir}/part3_normalized_cost.png"); plt.close()


def plot_convergence(histories, labels, output_file, title="Convergence"):
    max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        if len(h)<max_len:
            padded.append(h + [h[-1]]*(max_len-len(h)))
        else:
            padded.append(h[:max_len])
    median_curve = np.median(np.array(padded),axis=0)
    x = list(range(1,max_len+1))
    plt.figure(figsize=(8,4))
    for h, lab in zip(histories, labels):
        plt.plot(range(1,len(h)+1), h, alpha=0.3, label=f"{lab}")
    plt.plot(x, median_curve, color='black', linewidth=2, label='median across runs')
    plt.xlabel('Iteration / Generation')
    plt.ylabel('Best-so-far Cost')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    matrices_sizes = {}
    for a in [5, 6, 7, 8, 9, 10, 15]:
        matrices = []
        for b in range(4):
            MAT = np.loadtxt(f"test-data/{a}_random_adj_mat_{b}.txt")
            matrices.append(MAT)
        matrices_sizes[a] = matrices

    sweep_size = max(matrices_sizes.keys())
    mats_for_sweep = matrices_sizes[sweep_size]
    hc_restarts = [1,2,5,10]
    hc_medians = {}
    for r in hc_restarts:
        scores = []
        for mat in mats_for_sweep:
            _, cost, _ = hill_climbing_random_restarts(mat, num_restarts=r, inner_iters=500)
            scores.append(cost)
        hc_medians[r] = median(scores)
    plt.figure(); plt.plot(list(hc_medians.keys()), list(hc_medians.values()), marker='o')
    plt.xlabel('num_restarts'); plt.ylabel('Median Tour Cost'); plt.title('Hill Climbing: cost vs num_restarts')
    plt.grid(True); plt.tight_layout()
    plt.savefig('experiment3-images/hc_hyperparam.png'); plt.close()

    hc_histories = []
    for i in range(5):
        _, _, hist = hill_climbing_random_restarts(mats_for_sweep[0], num_restarts=5, inner_iters=500)
        hc_histories.append(hist)
    plot_convergence(hc_histories, [f"run{i+1}" for i in range(len(hc_histories))],
                     output_file='experiment3-images/hc_convergence.png', title='Hill Climbing Convergence')

    sa_histories = []
    for i in range(5):
        _, _, hist = simulated_annealing(mats_for_sweep[0], T0=100.0, alpha=0.995, max_iters=500)
        sa_histories.append(hist)
    plot_convergence(sa_histories, [f"run{i+1}" for i in range(len(sa_histories))],
                     output_file='experiment3-images/sa_convergence.png', title='Simulated Annealing Convergence')

    ga_histories = []
    for i in range(5):
        _, _, hist = genetic_algorithm(mats_for_sweep[0], population_size=50, mutation_chance=0.1, num_generations=200)
        ga_histories.append(hist)
    plot_convergence(ga_histories, [f"run{i+1}" for i in range(len(ga_histories))],
                     output_file='experiment3-images/ga_convergence.png', title='Genetic Algorithm Convergence')

    sa_T0_results, sa_alpha_results = hyperparameter_sweep_sa(mats_for_sweep, T0_values=[10,50,100], alpha_values=[0.99,0.995,0.999], max_iters=500)
    plt.figure(); plt.plot(list(sa_T0_results.keys()), list(sa_T0_results.values()), marker='o'); plt.title("SA: median cost vs T0"); plt.savefig('experiment3-images/sa_T0_sweep.png'); plt.close()
    plt.figure(); plt.plot(list(sa_alpha_results.keys()), list(sa_alpha_results.values()), marker='o'); plt.title("SA: median cost vs alpha"); plt.savefig('experiment3-images/sa_alpha_sweep.png'); plt.close()

    ga_pop_results, ga_mut_results = hyperparameter_sweep_ga(mats_for_sweep, pop_sizes=[20,50,100], mutation_chances=[0.01,0.05,0.1], num_generations=200)
    plt.figure(); plt.plot(list(ga_pop_results.keys()), list(ga_pop_results.values()), marker='o'); plt.title("GA: median cost vs pop_size"); plt.savefig('experiment3-images/ga_pop_sweep.png'); plt.close()
    plt.figure(); plt.plot(list(ga_mut_results.keys()), list(ga_mut_results.values()), marker='o'); plt.title("GA: median cost vs mutation"); plt.savefig('experiment3-images/ga_mut_sweep.png'); plt.close()

    stats_part3 = compare_part3_algorithms(
    matrices_sizes,
    hc_restarts=5,
    sa_params={'T0':100,'alpha':0.995,'max_iters':500},
    ga_params={'population_size':50,'mutation_chance':0.1,'num_generations':200},
    astar_func=astar_tsp  
    )
    plot_normalized_part3(stats_part3)