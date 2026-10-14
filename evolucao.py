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
MAX_HOURS = 2
NUM_EPISODES = 6
MAX_STEPS = 10000
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05
NUM_ELITE = 2
TOURNAMENT_SIZE = 3

VALIDATE_EVERY = 25
NUM_FINALISTS = 5
VALIDATION_EPISODES = 50
VALIDATION_SEED = 1_000_000

RESUME = True

SEED = 42
RESULTS_DIR = "results"
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.csv")
BEST_AGENT_PATH = os.path.join(RESULTS_DIR, "best_agent.npz")


def evaluate(agent, num_episodes=NUM_EPISODES, seed=None):
    env = create_env()
    total_reward = 0.0

    for episode in range(num_episodes):

        episode_seed = None if seed is None else seed + episode
        state, info = env.reset(seed=episode_seed)
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


def create_population(start_agent=None):

    population = []

    if start_agent is not None:
        population.append(copy.deepcopy(start_agent))

    while len(population) < POPULATION_SIZE:
        if start_agent is None:
            new_agent = Agent()
        else:
            new_agent = copy.deepcopy(start_agent)
            mutate(new_agent)
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


def evaluate_population(population, pool, seed):
    scores = pool.starmap(evaluate, [(agent, NUM_EPISODES, seed) for agent in population])
    for agent, score in zip(population, scores):
        agent.fitness = score


def save_history(history):
    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def load_history():
    with open(HISTORY_PATH) as file:
        rows = list(csv.DictReader(file))

    return [
        {
            "generation": int(row["generation"]),
            "best": float(row["best"]),
            "mean": float(row["mean"]),
            "worst": float(row["worst"]),
        }
        for row in rows
    ]


def validation_scores(agents, pool):
    games = [
        (agent, 1, VALIDATION_SEED + episode)
        for agent in agents
        for episode in range(VALIDATION_EPISODES)
    ]
    results = np.array(pool.starmap(evaluate, games))
    return results.reshape(len(agents), VALIDATION_EPISODES).mean(axis=1)


def validate(population, pool, best_validated):
    finalists = sorted(population, key=lambda agent: agent.fitness, reverse=True)[:NUM_FINALISTS]

    averages = validation_scores(finalists, pool)

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

    start = time.time()
    deadline = start + MAX_HOURS * 3600

    with Pool() as pool:
        if RESUME and os.path.exists(BEST_AGENT_PATH):
            start_agent = load_agent(BEST_AGENT_PATH)
            history = load_history()
            best_validated = float(validation_scores([start_agent], pool)[0])
            print(
                f"Resuming from {BEST_AGENT_PATH} at generation {len(history)}: "
                f"{best_validated:.1f} points",
                flush=True,
            )
        else:
            start_agent = None
            history = []
            best_validated = float("-inf")

        population = create_population(start_agent)

        for generation in range(len(history), NUM_GENERATIONS):
            evaluate_population(population, pool, seed=generation * NUM_EPISODES)

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
