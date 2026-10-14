import csv
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

from asteroids_neuroevolucao import NUM_ACTIONS, load_agent
from evolucao import BEST_AGENT_PATH, RESULTS_DIR, evaluate

TEST_EPISODES = 100
TEST_SEED = 2_000_000
BASELINES_PATH = os.path.join(RESULTS_DIR, "baselines.csv")

RIGHT = 3
LEFT = 4
RIGHTFIRE = 9
LEFTFIRE = 10


class RandomAgent:
    def reset(self):
        pass

    def act(self, state):
        return random.randrange(NUM_ACTIONS)


class SpinFireAgent:
    def __init__(self, fire_action, turn_action):
        self.fire_action = fire_action
        self.turn_action = turn_action
        self.firing = False

    def reset(self):
        self.firing = False

    def act(self, state):
        self.firing = not self.firing
        return self.fire_action if self.firing else self.turn_action


def test_scores(agent, pool):
    games = [(agent, 1, TEST_SEED + episode) for episode in range(TEST_EPISODES)]
    return np.array(pool.starmap(evaluate, games))


if __name__ == "__main__":
    agent_path = sys.argv[1] if len(sys.argv) > 1 else BEST_AGENT_PATH

    agents = {
        "random": RandomAgent(),
        "spin_left_fire": SpinFireAgent(LEFTFIRE, LEFT),
        "spin_right_fire": SpinFireAgent(RIGHTFIRE, RIGHT),
        "evolved_agent": load_agent(agent_path),
    }

    rows = []
    with Pool() as pool:
        for name, agent in agents.items():
            scores = test_scores(agent, pool)
            mean = float(scores.mean())
            error = float(scores.std() / np.sqrt(len(scores)))
            rows.append({"agent": name, "mean": mean, "std_error": error})
            print(f"{name:16s} {mean:7.1f} ± {error:5.1f}  ({TEST_EPISODES} games)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BASELINES_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["agent", "mean", "std_error"])
        writer.writeheader()
        writer.writerows(rows)
