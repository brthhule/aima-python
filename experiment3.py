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

# -----------------------
# Ensure output folder exists
# -----------------------
os.makedirs("experiment3-images", exist_ok=True)

# -----------------------
# Helper functions
# -----------------------
def calculate_cost(graph, tour):
    return sum(graph[tour[i]][tour[i+1]] for i in range(len(tour)-1))

def time_function(func, *args, **kwargs):
    t0 = time.time_ns()
    cpu0 = time.process_time_ns()
    result = func(*args, **kwargs)
    t1 = time.time_ns()
    cpu1 = time.process_time_ns()
    return t1 - t0, cpu1 - cpu0, result

# -----------------------
# Hill Climbing
# -----------------------
def hill_climbing_random_restarts(graph, num_restarts=10, inner_iters=1000):
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
        # Merge histories
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

# -----------------------
# Simulated Annealing
# -----------------------
def simulated_annealing(graph, T0=100.0, alpha=0.995, max_iters=2000):
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

# -----------------------
# Genetic Algorithm
# -----------------------
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

# -----------------------
# Convergence plotting
# -----------------------
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

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Load matrices
    matrices_sizes = {}
    for a in [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]:
        matrices = []
        for b in range(4):
            MAT = np.loadtxt(f"test-data/{a}_random_adj_mat_{b}.txt")
            matrices.append(MAT)
        matrices_sizes[a] = matrices

    # -----------------------
    # Hyperparameter sweep example: Hill Climbing
    # -----------------------
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

    # -----------------------
    # Convergence example: Hill Climbing
    # -----------------------
    hc_histories = []
    for i in range(5):
        _, _, hist = hill_climbing_random_restarts(mats_for_sweep[0], num_restarts=5, inner_iters=500)
        hc_histories.append(hist)
    plot_convergence(hc_histories, [f"run{i+1}" for i in range(len(hc_histories))],
                     output_file='experiment3-images/hc_convergence.png', title='Hill Climbing Convergence')

    # -----------------------
    # Convergence example: Simulated Annealing
    # -----------------------
    sa_histories = []
    for i in range(5):
        _, _, hist = simulated_annealing(mats_for_sweep[0], T0=100.0, alpha=0.995, max_iters=500)
        sa_histories.append(hist)
    plot_convergence(sa_histories, [f"run{i+1}" for i in range(len(sa_histories))],
                     output_file='experiment3-images/sa_convergence.png', title='Simulated Annealing Convergence')

    # -----------------------
    # Convergence example: Genetic Algorithm
    # -----------------------
    ga_histories = []
    for i in range(5):
        _, _, hist = genetic_algorithm(mats_for_sweep[0], population_size=50, mutation_chance=0.1, num_generations=200)
        ga_histories.append(hist)
    plot_convergence(ga_histories, [f"run{i+1}" for i in range(len(ga_histories))],
                     output_file='experiment3-images/ga_convergence.png', title='Genetic Algorithm Convergence')
