import copy
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# ---------------------------------------------------------
# ENVIRONMENT AND NETWORK SETTINGS
# ---------------------------------------------------------

ENV_ID = "ALE/Asteroids-v5"

NUM_RAM = 128
NUM_ACTIONS = 14

NUM_INPUTS = NUM_RAM + NUM_ACTIONS
NUM_HIDDEN = 32


def create_env(render_mode=None):
    # frameskip=4 and sticky actions
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode
    )


class Agent:
    def __init__(self):
        self.W1 = np.random.randn(NUM_INPUTS, NUM_HIDDEN) / np.sqrt(NUM_INPUTS)
        self.b1 = np.zeros(NUM_HIDDEN)
        self.W2 = np.random.randn(NUM_HIDDEN, NUM_ACTIONS) / np.sqrt(NUM_HIDDEN)
        self.b2 = np.zeros(NUM_ACTIONS)
        self.fitness = 0.0
        self.prev_action = 0

    def reset(self):
        self.prev_action = 0

    def act(self, state):
        ram = state.astype(np.float32) / 255.0

        prev_one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
        prev_one_hot[self.prev_action] = 1.0

        inputs = np.concatenate([ram, prev_one_hot])

        hidden = np.tanh(inputs @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2

        action = int(np.argmax(output))
        self.prev_action = action
        return action

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)


def load_agent(path):
    weights = np.load(path)
    agent = Agent()
    agent.W1 = weights["W1"]
    agent.b1 = weights["b1"]
    agent.W2 = weights["W2"]
    agent.b2 = weights["b2"]
    return agent
