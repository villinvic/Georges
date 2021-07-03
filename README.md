# GEORGES

### Generating Evolutionary Opponents as a Reinforcement Guided Exploration Solution

GEORGES experimentation on Super Smash Bros Melee , in 2 vs 2 official 
competition standards.

### Description

GEORGES is an Evolutionary Reinforcement Learning Framework inspired by emerging Evolutionary
solutions **[1, 2]** which are leveraging from a Population of individuals, allowing parameter
and reward auto-tuning.
 
GEORGES is a combination of Population Based Training, Further Genetic operations
(Mutation, Crossover...) and Tournament Simulation (Pool and Bracket model).

We train a population of individuals, or players, where each player mains a character and tries to
maximize their Elo score.
An individual with a score too low will eventually be replaced by a mutated version of
a higher ranked individual.

The winning team of a tournament will generate an offspring through crossover, which
will replace the worst player Elo-wise in the population.

The V-Trace algorithm **[3]** (an Actor Critic algorithm variant with off-policy correction) is used
to train individuals, continuously, on top of the GA operators, from generated experience of
games simulated on [Dolphin](https://github.com/dolphin-emu/dolphin).

### Current Project status

In developpement,
Testing phase starting soon.

### Live demonstration

TBA

## References

**[1]** Shen, Ruimin, Yan Zheng, Jianye Hao, Zhaopeng Meng, Yingfeng Chen, Changjie Fan, and Yang Liu. “Generating Behavior-Diverse Game AIs with Evolutionary Multi-Objective Deep Reinforcement Learning.” In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 3371–77. Yokohama, Japan: International Joint Conferences on Artificial Intelligence Organization, 2020. https://doi.org/10.24963/ijcai.2020/466.

**[2]** Jaderberg, Max, Wojciech M. Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castañeda, Charles Beattie, et al. “Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning.” Science 364, no. 6443 (May 31, 2019): 859–65. https://doi.org/10.1126/science.aau6249.

**[3]** Espeholt, Lasse, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, et al. “IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.” ArXiv:1802.01561 [Cs], June 28, 2018. http://arxiv.org/abs/1802.01561.


#### Further notes

cf. Original zmq-exi Dolphin for speedhacking, by [Vladfi1](https://github.com/vladfi1/dolphin/tree/new-zmq-exi).

cf. [p3](https://github.com/spxtr/p3) for Original Dolphin<=>Python communication system.