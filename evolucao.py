import copy
import csv
import os
import random
import time
from multiprocessing import Pool

from asteroids_neuroevolucao import *

# ---------------------------------------------------------
# EVOLUTION SETTINGS
# ---------------------------------------------------------

POPULATION_SIZE = 40
NUM_GENERATIONS = 100000
MAX_HOURS = 8
NUM_EPISODES = 6
MAX_STEPS = 10000
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05
NUM_ELITE = 2
TOURNAMENT_SIZE = 3

VALIDATE_EVERY = 25
NUM_FINALISTS = 5
VALIDATION_EPISODES = 30

SEED = 42
RESULTS_DIR = "results"
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.csv")
BEST_AGENT_PATH = os.path.join(RESULTS_DIR, "best_agent.npz")


def evaluate(agent, num_episodes=NUM_EPISODES):
    env = create_env()
    total_reward = 0.0

    for _ in range(num_episodes):

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
    return total_reward / num_episodes


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


def save_history(history):
    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def validate(population, pool, best_validated):
    finalists = sorted(population, key=lambda agent: agent.fitness, reverse=True)[:NUM_FINALISTS]

    games = [(agent, 1) for agent in finalists for _ in range(VALIDATION_EPISODES)]
    results = np.array(pool.starmap(evaluate, games))
    averages = results.reshape(len(finalists), VALIDATION_EPISODES).mean(axis=1)

    winner = int(np.argmax(averages))
    print(
        f"  Validation ({VALIDATION_EPISODES} games each): "
        + ", ".join(f"{score:.0f}" for score in averages),
        flush=True,
    )

    if averages[winner] > best_validated:
        best_validated = float(averages[winner])
        finalists[winner].save(BEST_AGENT_PATH)
        print(f"  New best agent saved: {best_validated:.1f} points", flush=True)

    return best_validated


def train():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    population = create_population()
    history = []
    best_validated = float("-inf")
    start = time.time()
    deadline = start + MAX_HOURS * 3600

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
            save_history(history)

            print(
                f"Generation {generation:04d} | "
                f"Best: {best:7.1f} | "
                f"Mean: {mean:7.1f} | "
                f"Worst: {worst:7.1f} | "
                f"{(time.time() - start) / 3600:5.2f}h",
                flush=True,
            )

            is_last = generation == NUM_GENERATIONS - 1 or time.time() >= deadline

            if (generation + 1) % VALIDATE_EVERY == 0 or is_last:
                best_validated = validate(population, pool, best_validated)

            if is_last:
                break

            population = new_generation(population)

    print(f"\nDone. Best agent: {best_validated:.1f} points, saved to {BEST_AGENT_PATH}")
    return population, history


if __name__ == "__main__":
    train()
