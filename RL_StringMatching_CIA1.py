#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:42:05 2024

@author: jaishree
"""
import random
import numpy as np

# Define target string and initial parameters
TARGET_STRING = "HELLO"
POPULATION_SIZE = 20
GENERATIONS = 150
K = 3  # Number of arms for mutation rate, crossover, and selection strategy
REWARD_DISCOUNT = 0.9  # Discount for previous rewards in K-arm bandit

# Define bandit arms and initial Q-values for each
arms = {
    'mutation_rate': [0.01, 0.05, 0.1],
    'crossover_type': ['single', 'multi'],
    'selection_strategy': ['roulette', 'tournament']
}
Q_values = {arm: [0] * len(arms[arm]) for arm in arms}

# Initialize reward trackers for arms
reward_counts = {arm: [0] * len(arms[arm]) for arm in arms}

# Function to initialize the population
def initialize_population():
    population = [''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=len(TARGET_STRING)))
                  for _ in range(POPULATION_SIZE)]
    return population

# Fitness function to calculate fitness of each individual
def calculate_fitness(individual):
    return sum(1 for i, j in zip(individual, TARGET_STRING) if i == j)

# Selection based on selection strategy chosen by the K-arm bandit
def select_parents(population, fitness_scores, strategy):
    if strategy == 'roulette':
        total_fitness = sum(fitness_scores)
        probs = [f / total_fitness for f in fitness_scores]
        return np.random.choice(population, size=2, p=probs)
    elif strategy == 'tournament':
        participants = random.sample(population, 4)
        participants.sort(key=calculate_fitness, reverse=True)
        return participants[:2]

# Crossover function based on chosen crossover type
def crossover(parent1, parent2, crossover_type):
    if crossover_type == 'single':
        point = random.randint(1, len(TARGET_STRING) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    elif crossover_type == 'multi':
        point1, point2 = sorted(random.sample(range(1, len(TARGET_STRING)), 2))
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return offspring1, offspring2

# Mutation function
def mutate(individual, mutation_rate):
    return ''.join(
        (char if random.random() > mutation_rate else random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for char in individual
    )

# Function to choose an action based on Q-values and perform exploration-exploitation
def select_arm(arm_name):
    arm_Q_values = Q_values[arm_name]
    max_Q = max(arm_Q_values)
    best_arms = [i for i, q in enumerate(arm_Q_values) if q == max_Q]
    chosen_arm = random.choice(best_arms) if random.random() < 0.9 else random.randint(0, len(arm_Q_values) - 1)
    return chosen_arm

# Update Q-values based on received reward
def update_q_values(arm_name, arm_index, reward):
    count = reward_counts[arm_name][arm_index]
    reward_counts[arm_name][arm_index] += 1
    learning_rate = 1 / (count + 1)
    Q_values[arm_name][arm_index] = (1 - learning_rate) * Q_values[arm_name][arm_index] + learning_rate * reward

# Genetic algorithm loop
def genetic_algorithm():
    population = initialize_population()
    best_individual = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        avg_fitness = sum(fitness_scores) / POPULATION_SIZE
        print(f"Generation {generation + 1}: Avg Fitness = {avg_fitness}")

        # Choose arms using K-arm bandit
        mutation_arm = select_arm('mutation_rate')
        crossover_arm = select_arm('crossover_type')
        selection_arm = select_arm('selection_strategy')

        mutation_rate = arms['mutation_rate'][mutation_arm]
        crossover_type = arms['crossover_type'][crossover_arm]
        selection_strategy = arms['selection_strategy'][selection_arm]

        # Generate new population
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness_scores, selection_strategy)
            offspring1, offspring2 = crossover(parent1, parent2, crossover_type)
            new_population.append(mutate(offspring1, mutation_rate))
            new_population.append(mutate(offspring2, mutation_rate))

        population = new_population
        best_fitness_in_gen = max(fitness_scores)
        
        if best_fitness_in_gen > best_fitness:
            best_fitness = best_fitness_in_gen
            best_individual = population[fitness_scores.index(best_fitness_in_gen)]
        
        # Reward based on fitness improvement
        reward = (best_fitness_in_gen - avg_fitness) / len(TARGET_STRING)
        update_q_values('mutation_rate', mutation_arm, reward)
        update_q_values('crossover_type', crossover_arm, reward)
        update_q_values('selection_strategy', selection_arm, reward)

        if best_fitness == len(TARGET_STRING):
            print(f"Solution found in Generation {generation + 1}: {best_individual}")
            break

    return best_individual, best_fitness

# Run genetic algorithm with K-arm bandit approach
best_solution, fitness = genetic_algorithm()
print(f"Best Individual: {best_solution}")
print(f"Best Fitness: {fitness}")
