#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:51:51 2024

@author: jaishree
"""

import matplotlib.pyplot as plt
from grid_world import GridWorld, AgentDP, AgentQLearning, AgentSARSA

def visualize_policy(policy):
    plt.figure(figsize=(10, 10))
    plt.imshow(policy, cmap='jet', interpolation='nearest')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Actions')
    plt.title('DP Policy Visualization')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()

def main():
    gridworld = GridWorld(size=100, obstacles_ratio=0.2)

    # Dynamic Programming Agent
    agent_dp = AgentDP(gridworld)
    agent_dp.value_iteration()
    print("DP Policy:")
    print(agent_dp.policy)
    visualize_policy(agent_dp.policy)

    # Q-learning Agent
    agent_q_learning = AgentQLearning(gridworld)
    agent_q_learning.train(episodes=10000)
    print("Q-learning Q-table:")
    print(agent_q_learning.q_table)

    # SARSA Agent
    agent_sarsa = AgentSARSA(gridworld)
    agent_sarsa.train(episodes=10000)
    print("SARSA Q-table:")
    print(agent_sarsa.q_table)

if __name__ == "__main__":
    main()

