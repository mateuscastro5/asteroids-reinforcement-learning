import copy
import random

from asteroids_neuroevolucao import *

# ---------------------------------------------------------
# EVOLUTION SETTINGS
# ---------------------------------------------------------

POPULATION_SIZE = 40
NUM_GENERATIONS = 50
NUM_EPISODES = 6
MAX_STEPS = 3000
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05
NUM_ELITE = 2
TOURNAMENT_SIZE = 3


def evaluate(agent):
    env = create_env()
    total_reward = 0.0

    for _ in range(NUM_EPISODES):

        state, info = env.reset()
        agent.reset()
        truncated = False
        terminated = False
        steps = 0
        score = 0.0

        while not (terminated or truncated) and steps < MAX_STEPS:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            steps += 1

        total_reward += score

    env.close()
    return total_reward / NUM_EPISODES


def create_population():

    population = []

    for _ in range(POPULATION_SIZE):
        new_agent = Agent()
        population.append(new_agent)

    return population


def mutate(agent):
    for table in [agent.W1, agent.b1, agent.W2, agent.b2]:
        mask = np.random.rand(*table.shape) < MUTATION_RATE
        noise = np.random.randn(*table.shape) * MUTATION_STRENGTH
        table += noise * mask


def select(population):
    # TODO (Step 4)
    # draw TOURNAMENT_SIZE random agents
    # return the one with the highest fitness
    pass


def new_generation(population):
    # TODO (Step 5)
    # sort the population by fitness (best first)
    # copy the NUM_ELITE best ones unchanged
    # fill the rest with children: parent = select, child = copy of parent, mutate child
    # return the new list with POPULATION_SIZE agents
    pass


if __name__ == "__main__":
    a = Agent()
    before_W1 = a.W1.copy()
    before_b1 = a.b1.copy()
    mutate(a)
    print((a.W1 != before_W1).sum(), "of", a.W1.size, "W1 weights changed")
    print((a.b1 != before_b1).sum(), "of", a.b1.size, "b1 weights changed")
