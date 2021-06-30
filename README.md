# GEORGES

###Generating Evolutionary Opponents as a Reinforcement Guided Exploration Solution

GEORGES experimentation on Super Smash Bros Melee , in  a 2 vs 2 official 
competition standard.

### Description

GEORGES is an Evolutionary Reinforcement Learning Framework inspired by emerging Evolutionary
solutions which are leveraging from a Population of individuals, allowing parameter
and reward auto-tuning.
 
GEORGES is a combination of Population Based Training, Further Genetic operations
(Mutation, Crossover...) and Tournament Simulation (Pool and Bracket model).

We train a population of individuals, or players, where each player mains a character and tries to
maximize their Elo score.
An individual with a score too low will eventually be replaced by a mutated version of
a higher ranked individual.

The winning team of a tournament will generate an offspring through crossover, which
will replace the worst player Elo-wise in the population.

### Current Project status

In developpement,
Testing phase starting soon.

### Live demonstration

TBA