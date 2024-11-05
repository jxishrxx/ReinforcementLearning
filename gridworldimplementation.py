#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:12:05 2024

@author: jaishree
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=100, start=None, goal=None, obstacles_ratio=0.2):
        self.size = size
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (size-1, size-1)
        self.grid = np.zeros((size, size))
        self.obstacles_ratio = obstacles_ratio
        self._place_obstacles()

    def _place_obstacles(self):
        num_obstacles = int(self.size ** 2 * self.obstacles_ratio)
        obstacles = np.random.choice(self.size * self.size, num_obstacles, replace=False)
        for idx in obstacles:
            x, y = divmod(idx, self.size)
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = -1  # Mark as obstacle

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.goal

    def step(self, state, action):
        if self.is_terminal(state):
            return state, 0, True

        next_state = self._move(state, action)

        if self.grid[next_state] == -1:  # hit an obstacle
            next_state = state  # stay in the same place
            reward = -5  # higher penalty for hitting an obstacle
        elif self.is_terminal(next_state):
            reward = 20  # higher reward for reaching the goal
        else:
            reward = -0.1  # small penalty for each move

        return next_state, reward, self.is_terminal(next_state)

    def _move(self, state, action):
        x, y = state
        if action == 0:   # up
            return (max(0, x - 1), y)
        elif action == 1: # down
            return (min(self.size - 1, x + 1), y)
        elif action == 2: # left
            return (x, max(0, y - 1))
        elif action == 3: # right
            return (x, min(self.size - 1, y + 1))

class AgentMDP:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.value_table = np.zeros((gridworld.size, gridworld.size))
        self.policy = np.zeros((gridworld.size, gridworld.size), dtype=int)
        self.gamma = 0.9
        self.epsilon = 1e-5
        self.actions = [0, 1, 2, 3]  # up, down, left, right

    def value_iteration(self):
        while True:
            delta = 0
            for x in range(self.gridworld.size):
                for y in range(self.gridworld.size):
                    state = (x, y)
                    if self.gridworld.is_terminal(state) or self.gridworld.grid[x, y] == -1:
                        continue
                    v = self.value_table[x, y]
                    self.value_table[x, y] = max(self._calculate_value(state, action) for action in self.actions)
                    delta = max(delta, abs(v - self.value_table[x, y]))
            if delta < self.epsilon:
                break
        self._extract_policy()

    def _calculate_value(self, state, action):
        next_state, reward, _ = self.gridworld.step(state, action)
        return reward + self.gamma * self.value_table[next_state]

    def _extract_policy(self):
        for x in range(self.gridworld.size):
            for y in range(self.gridworld.size):
                state = (x, y)
                if self.gridworld.is_terminal(state) or self.gridworld.grid[x, y] == -1:
                    continue
                self.policy[x, y] = np.argmax([self._calculate_value(state, action) for action in self.actions])

class AgentQLearning:
    def __init__(self, gridworld, learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.gridworld = gridworld
        self.q_table = np.zeros((gridworld.size, gridworld.size, 4))  # 4 actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

    def train(self, episodes):
        for _ in range(episodes):
            state = self.gridworld.reset()
            while not self.gridworld.is_terminal(state):
                if random.random() < self.epsilon:
                    action = random.choice(range(4))  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit
                next_state, reward, _ = self.gridworld.step(state, action)
                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[state][action] += self.learning_rate * (reward + self.discount * self.q_table[next_state][best_next_action] - self.q_table[state][action])
                state = next_state

def visualize_policy(policy):
    direction = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    grid_size = policy.shape[0]
    visual_grid = np.full((grid_size, grid_size), ' ')
    for x in range(grid_size):
        for y in range(grid_size):
            if policy[x, y] != -1:  # if not an obstacle
                visual_grid[x, y] = direction[policy[x, y]]
    plt.imshow(np.zeros((grid_size, grid_size)), cmap='Greys', vmin=-1, vmax=1)
    for (i, j), val in np.ndenumerate(visual_grid):
        plt.text(j, i, val, ha='center', va='center')
    plt.title('Optimal Policy Visualization')
    plt.show()

def main():
    grid_size = 100
    gridworld = GridWorld(size=grid_size, obstacles_ratio=0.2)
    
    # MDP Agent
    mdp_agent = AgentMDP(gridworld)
    mdp_agent.value_iteration()
    
    # Q-Learning Agent
    q_learning_agent = AgentQLearning(gridworld)
    q_learning_agent.train(episodes=10000)
    
    # Visualize policies
    visualize_policy(mdp_agent.policy)
    print("MDP Policy Value Table:\n", mdp_agent.value_table)
    print("Q-Learning Q-Table Sample:\n", q_learning_agent.q_table[50:60, 50:60])

if __name__ == "__main__":
    main()
