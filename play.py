import sys

from asteroids_neuroevolucao import create_env, load_agent
from evolucao import BEST_AGENT_PATH


def play(num_games):
    agent = load_agent(BEST_AGENT_PATH)
    env = create_env(render_mode="human")

    for game in range(num_games):
        state, info = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        score = 0.0

        while not (terminated or truncated):
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward

        print(f"Game {game + 1}: {score:.0f} points")

    env.close()


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    play(num_games)
