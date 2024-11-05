# ReinforcementLearning

K-arm bandit-based solution for genetic algorithm
String matching

For a string matching problem, RL K-arm bandit based solution can be used accordingly with suitable parameters to solve the problem. Based on the problem, we choose state, action, reward and policy.
State: Current set of strings
Action: Selecting which string to mutate and what selection strategy can be applied
Reward: Increase in fitness score and overall accuracy
Policy: the system learns the best approach and strategy to increase the accuracy of the overall problem
Using RL could be efficient than a traditional genetic algorithm as it would reduce the random selection of strings and use a reward based system to find the most suitable values.

Incorporating K-arm bandit based solution:
For using a k-arm bandit based solution, we first define what arms do we use for what actions/strategies. The arms can be:
Mutation rate arm: For various mutation rates like low and high
Crossover type arm: Single point or multi point crossover can be given
Selection strategy arm: Any selection strategy like random selection, roulette wheel or tournament selection can be used.
Once the arms are defined, any one arm is used to proceed further. For instance, any particular mutation rate or selection strategy is chosen. Based on the selection strategy, we evolve the population.
Once the strategy is chosen, we define reward system which measures the improvement and checks how close it is to the target. Based on the closeness to desired output, the reward is given. Then this reward is used to update the value estimate. 
We set up the system in such a way that it uses both exploration and exploitation. The system should try different strategies to know about the effectiveness of them. Exploitation should also be done such that it chooses most of the strategies that have given previously best rewards.
Over time, the k arm bandit system helps in selecting most effective strategies for string matching that gives the most reliable and closer string to the target string faster.

Initially the system randomly selects a strategy. If the fitness of the arm is better then it is favoured in the next generation, if not then it is less favoured. Over time, it learns the best combination which works best and hence chosen for the further generations. By this way both exploration and exploitation are balanced.



