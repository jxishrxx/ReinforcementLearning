# ReinforcementLearning

# K-arm bandit-based solution for genetic algorithm
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


# MDP-based Reinforcement Learning Approach for Grid Pathfinding

In a grid pathfinding problem with obstacles, implementing reinforcement learning (RL) algorithms provides an effective solution for optimizing paths from a designated start to a goal. The Markov Decision Process (MDP) framework enables the definition of states, actions, rewards, and policies to guide an agent in making intelligent decisions to reach the goal efficiently. In this setup, the state space represents each cell in the grid, and actions include possible movements (up, down, left, right). Rewards are given for reaching the goal or avoiding obstacles, with small penalties applied to each step to encourage the shortest path. By progressively updating policies through learned experiences, the RL agent efficiently navigates toward the optimal path, reducing random exploration and adapting to obstacles within the grid environment.

The two versions of the MDP-based grid pathfinding code showcase different approaches to structuring the solution. The first code is well-organized and modular, with separate classes for the grid environment and each agent (dynamic programming, Q-learning, and SARSA). This clear separation makes it easier to understand how each component works and allows for easier modifications or testing of individual agents. If you want to tweak the behavior of a specific agent or add new functionality, this structure makes it straightforward. The clarity of this approach is particularly helpful for anyone looking to build upon the work or for new team members trying to understand the system.

On the other hand, the second code is more compact, combining the grid and agents into a tighter structure. While this can make the code look simpler and reduce some redundancy, it can also make it harder to follow and modify. If you need to change how one of the agents behaves, it may require digging through a more complex codebase, which can be a hassle.

In summary, while the second version might save some lines of code, the first version's modular approach is likely to be more beneficial in the long run, especially for larger projects or for those who plan to evolve the solution over time. Its clear layout promotes better understanding and easier enhancements, making it the preferred choice for collaborative work or ongoing development in reinforcement learning.

Ouput of first code:
MDP Policy Value Table:
 [[-0.99999956 -0.99999956  0.         ... -0.9998724  -0.99985822
  -0.9998724 ]
 [-0.99999956 -0.99999956  0.         ... -0.99985822 -0.99984246
   0.        ]
 [-0.99999956 -0.99999956 -0.99999956 ...  0.         -0.99982496
  -0.99980551]
 ...
 [-0.9998724  -0.99985822 -0.99984246 ... 14.309      16.01
  14.309     ]
 [-0.99988516 -0.9998724  -0.99985822 ...  0.         17.9
   0.        ]
 [-0.9998724  -0.99985822 -0.99984246 ... 17.9        20.
   0.        ]]
Q-Learning Q-Table Sample:
 [[[-5.68828816 -5.77360989 -0.9981375  -0.99813478]
  [-0.99799245 -0.99798744 -0.99801094 -0.99798756]
  [-0.99792442 -0.99792799 -0.99791999 -5.85020241]
  [ 0.          0.          0.          0.        ]
  [-0.9976654  -5.71230997 -5.65969156 -0.99764355]
  [-0.9975537  -0.99754145 -0.9975357  -0.99752983]
  [-0.9974204  -0.99740947 -0.99741473 -0.99741341]
  [-0.99727837 -0.99725568 -0.99730092 -0.99726903]
  [-0.99714037 -0.99713412 -0.99714304 -5.79762154]
  [ 0.          0.          0.          0.        ]]

 [[ 0.          0.          0.          0.        ]
  [-0.99784961 -0.99785201 -5.87135937 -0.99786437]
  [-0.99781666 -0.99781824 -0.99781997 -5.78320515]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [-0.99743441 -5.63990143 -5.77134224 -0.99743659]
  [-0.99730249 -5.88947241 -0.99732934 -0.99730711]
  [-0.99714412 -0.99712261 -0.9971498  -0.99711638]
  [-0.99693985 -0.99693449 -0.99701602 -0.99694095]
  [-5.83725069 -0.99674545 -0.99675755 -5.76450471]]

 [[-5.62353884 -0.99771987 -5.65921006 -0.99770266]
  [-0.99772778 -0.99773554 -0.99773049 -0.99772944]
  [-0.99769341 -0.99767113 -0.99769103 -0.99766339]
  [-5.85488787 -0.99753469 -0.99754302 -0.99753252]
  [-5.74593975 -0.99745148 -0.99744051 -5.35848884]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [-0.99695652 -0.99692908 -5.86138848 -0.99692363]
  [-0.99675849 -0.99673917 -0.99676823 -0.99674891]
  [-0.9965592  -0.99653153 -0.99654584 -0.99652088]]

 [[-0.99769677 -0.99767194 -5.81234509 -0.99767366]
  [-0.99764316 -0.99764372 -0.99764482 -0.99763885]
  [-0.99756094 -0.99757139 -0.99758551 -0.99757416]
  [-0.99746412 -5.86948539 -0.997493   -0.99747614]
  [-0.99737329 -0.99734185 -0.99734604 -0.99735236]
  [-5.81226684 -0.99721257 -0.99722065 -0.99720771]
  [-5.83359513 -0.99702944 -0.99710557 -0.99702829]
  [-0.99683795 -5.85136703 -0.99683123 -0.99681737]
  [-0.99658399 -0.99657118 -0.99655717 -0.99656232]
  [-0.9963268  -0.99631442 -0.99636499 -0.99632196]]

 [[-0.9976731  -5.86581345 -0.99767392 -0.99765657]
  [-0.99756286 -0.99758593 -0.99758258 -0.99755189]
  [-0.99749049 -0.99747482 -0.99749243 -5.80265502]
  [ 0.          0.          0.          0.        ]
  [-0.99726857 -0.99723727 -5.76517997 -0.99724824]
  [-0.99720312 -0.99717956 -0.9972092  -0.99718979]
  [-0.99710633 -0.99713703 -0.99713776 -5.79036192]
  [ 0.          0.          0.          0.        ]
  [-0.99633347 -5.50867988 -5.63501867 -0.99633708]
  [-0.99610607 -0.9960531  -0.99610672 -0.99606341]]

 [[ 0.          0.          0.          0.        ]
  [-0.99749195 -0.99749849 -5.87155951 -0.99748053]
  [-0.99737547 -5.85460006 -0.99735071 -0.99732157]
  [-5.79574887 -0.99709771 -0.99715028 -0.99711896]
  [-0.99719378 -5.83782545 -0.99716529 -0.99716808]
  [-0.99718518 -0.9971558  -0.99716651 -0.99716486]
  [-0.99715107 -5.74058133 -0.99715737 -5.48195184]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [-0.99576156 -0.99574297 -5.84072757 -0.99575588]]

 [[ 0.          0.          0.          0.        ]
  [-0.99754438 -0.99754616 -5.62701434 -5.65998308]
  [ 0.          0.          0.          0.        ]
  [-0.99689171 -0.99685435 -5.77148087 -5.72473224]
  [ 0.          0.          0.          0.        ]
  [-0.99716009 -4.69043808 -5.03317648 -4.32854011]
  [ 0.          0.          0.          0.        ]
  [-5.73771197 -0.99526158 -5.60882926 -0.9952521 ]
  [-5.52823281 -0.99525198 -0.99528823 -0.99526938]
  [-0.99543484 -0.99541433 -0.99540675 -5.74499498]]

 [[-5.60561607 -5.70405361 -0.99762563 -0.99763063]
  [-0.99759359 -5.22525435 -0.99758615 -5.58880754]
  [ 0.          0.          0.          0.        ]
  [-0.99665873 -0.99658636 -5.85125514 -0.99658807]
  [-5.79974275 -0.99635499 -0.99638667 -5.60756536]
  [ 0.          0.          0.          0.        ]
  [-5.40494758 -0.99534185 -5.56902636 -0.99531616]
  [-0.9952207  -0.99524795 -0.99522728 -0.99524565]
  [-0.99512916 -0.99512456 -0.99517839 -0.995142  ]
  [-0.99511517 -0.99509621 -0.9950801  -0.99507067]]

 [[ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [-0.99641466 -0.99635914 -5.79794444 -0.99636171]
  [-0.99613392 -5.87955443 -0.99615145 -0.99610148]
  [-5.8635829  -0.99576206 -0.99579815 -0.99574862]
  [-0.99542096 -0.99544011 -0.99545953 -0.99542543]
  [-0.99518363 -0.99516806 -0.99518907 -0.99516989]
  [-0.99497853 -0.99495123 -0.99498547 -0.99494039]
  [-0.99484395 -0.9947886  -0.99478073 -5.80292102]]

 [[-5.55167702 -0.9964572  -5.45991549 -0.99646991]
  [-5.8336465  -0.99639004 -0.99638091 -0.9964081 ]
  [-5.80554178 -0.99629568 -0.99633193 -0.99635915]
  [-0.99637475 -5.67936405 -0.99634332 -5.48926589]
  [ 0.          0.          0.          0.        ]
  [-0.9954981  -0.99549216 -5.74674543 -0.99547594]
  [-0.99525785 -0.99526167 -0.99525728 -0.99524283]
  [-0.9949768  -0.99497734 -0.9949963  -0.99497743]
  [-0.99474515 -0.99472458 -0.99470892 -0.99471076]
  [-0.99440209 -0.99440812 -0.99447255 -0.99440547]]]


Output of second code:
DP Policy:
[[0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 0]
 [0 1 0 ... 2 2 2]
 ...
 [0 0 0 ... 1 1 1]
 [0 0 3 ... 3 1 1]
 [0 0 0 ... 0 3 0]]
Q-learning Q-table:
[[[-0.99999133 -0.99999132 -0.99999135 -0.99999132]
  [-0.99999047 -1.89998849 -0.9999907  -0.99999044]
  [-0.99998969 -0.99998946 -0.99998957 -0.99998946]
  ...
  [-0.99840191 -0.99841435 -0.99841375 -1.87546183]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]]

 [[-0.9999906  -0.99999043 -0.99999043 -1.89998882]
  [ 0.          0.          0.          0.        ]
  [-0.99998845 -0.99998847 -1.89996247 -0.99998844]
  ...
  [-0.99840706 -0.99841737 -0.99842022 -0.99841136]
  [-1.80922938 -0.99840985 -0.99842026 -1.81707319]
  [ 0.          0.          0.          0.        ]]

 [[-0.99998952 -0.99998946 -0.99998959 -0.99998946]
  [-1.89997987 -0.99998847 -0.99998867 -0.99998845]
  [-0.99998763 -1.89994604 -0.99998782 -0.99998762]
  ...
  [-0.99840748 -1.8448332  -0.99840515 -0.99841617]
  [-0.99842198 -1.87434551 -0.99840796 -0.99841593]
  [-1.85833843 -0.99841948 -0.99842337 -0.99841341]]

 ...

 [[-0.99893184 -0.99894032 -0.99893679 -0.99893652]
  [-0.99892051 -0.99892489 -0.99894481 -1.87131569]
  [ 0.          0.          0.          0.        ]
  ...
  [-0.029701   -0.036991   -0.1         4.37526495]
  [ 2.80201864  5.86201711  1.42125118  7.91      ]
  [ 7.019       8.9         7.019       7.91      ]]

 [[-0.99893177 -0.99892749 -0.99894246 -0.99893431]
  [-0.99892892 -0.99892424 -0.99892264 -0.99893176]
  [-1.88017582 -0.99891791 -0.99892661 -0.99891793]
  ...
  [-0.0199     -0.1        -0.0199      3.22960044]
  [ 3.27694675  1.70301872  0.52844873  8.9       ]
  [ 7.91       10.          7.91        8.9       ]]

 [[-0.99893312 -0.99893212 -0.99892454 -0.99892734]
  [-0.99892303 -0.99892978 -0.99893006 -0.9989213 ]
  [-0.99891499 -0.99892282 -0.9989212  -0.99891949]
  ...
  [ 0.          0.          0.          0.        ]
  [ 0.76645533 -0.01       -0.1         5.6953279 ]
  [ 0.          0.          0.          0.        ]]]

  ![image](https://github.com/user-attachments/assets/9edf9954-125c-4bdd-86d4-243343046a69)

