import copy
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# ---------------------------------------------------------
# CONFIGURAÇÕES DO AMBIENTE E DA REDE
# ---------------------------------------------------------

ENV_ID = "ALE/Asteroids-v5"

NUM_RAM = 128
NUM_ACOES = 14

NUM_ENTRADAS = NUM_RAM + NUM_ACOES
NUM_OCULTOS = 32


def criar_ambiente(render_mode=None):
    # frameskip=4 e sticky actions
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode
    )


class Agente:
    def __init__(self):
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS) / np.sqrt(NUM_ENTRADAS)
        self.b1 = np.zeros(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES) / np.sqrt(NUM_OCULTOS)
        self.b2 = np.zeros(NUM_ACOES)
        self.fitness = 0.0
        self.acao_anterior = 0

    def reiniciar(self):
        self.acao_anterior = 0

    def agir(self, estado):
        ram = estado.astype(np.float32) / 255.0

        acao_ant = np.zeros(NUM_ACOES, dtype=np.float32)
        acao_ant[self.acao_anterior] = 1.0

        entrada = np.concatenate([ram, acao_ant])

        oculto = np.tanh(entrada @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2

        acao = int(np.argmax(saida))
        self.acao_anterior = acao
        return acao
