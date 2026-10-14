import copy
import random
import time
from multiprocessing import Pool

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

SEED = 42


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
    competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(competitors, key=lambda agent: agent.fitness)


def new_generation(population):
    population.sort(key=lambda agent: agent.fitness, reverse=True)

    new_population = []

    for i in range(NUM_ELITE):
        new_population.append(copy.deepcopy(population[i]))

    while len(new_population) < POPULATION_SIZE:
        parent = select(population)
        child = copy.deepcopy(parent)
        mutate(child)
        new_population.append(child)

    return new_population


def evaluate_population(population, pool):
    scores = pool.map(evaluate, population)
    for agent, score in zip(population, scores):
        agent.fitness = score


def train():
    random.seed(SEED)
    np.random.seed(SEED)

    population = create_population()
    history = []
    start = time.time()

    with Pool() as pool:
        for generation in range(NUM_GENERATIONS):
            evaluate_population(population, pool)

            scores = [agent.fitness for agent in population]
            best = max(scores)
            mean = float(np.mean(scores))
            worst = min(scores)

            history.append({
                "generation": generation,
                "best": best,
                "mean": mean,
                "worst": worst,
            })

            print(
                f"Generation {generation:03d} | "
                f"Best: {best:7.1f} | "
                f"Mean: {mean:7.1f} | "
                f"Worst: {worst:7.1f} | "
                f"{time.time() - start:5.0f}s"
            )

            if generation < NUM_GENERATIONS - 1:
                population = new_generation(population)

    return population, history


if __name__ == "__main__":
    train()
