import copy
import random

from asteroids_neuroevolucao import *

# ---------------------------------------------------------
# CONFIGURAÇÕES DA EVOLUÇÃO
# ---------------------------------------------------------

TAMANHO_POPULACAO = 40
NUM_GERACOES = 50
NUM_EPISODIOS = 6
MAX_PASSOS = 3000
TAXA_MUTACAO = 0.1
INTENSIDADE_MUTACAO = 0.05
NUM_ELITE = 2
TAMANHO_TORNEIO = 3


def avaliar(agente):
    env = criar_ambiente()
    estado, info = env.reset()
    print(estado)
    env.close()


def criar_populacao():
    # TODO (Passo 2)
    # devolver uma lista com TAMANHO_POPULACAO agentes novos
    pass


def mutar(agente):
    # TODO (Passo 3)
    # para cada tabela (W1, b1, W2, b2):
    #     sortear quais posições mudam (chance TAXA_MUTACAO)
    #     somar ruído × INTENSIDADE_MUTACAO nessas posições
    pass


def selecionar(populacao):
    # TODO (Passo 4)
    # sortear TAMANHO_TORNEIO agentes
    # devolver o de maior fitness
    pass


def nova_geracao(populacao):
    # TODO (Passo 5)
    # ordenar a população pela nota (maior primeiro)
    # copiar os NUM_ELITE melhores sem alteração
    # completar com filhos: pai = selecionar, filho = cópia do pai, mutar filho
    # devolver a nova lista com TAMANHO_POPULACAO agentes
    pass


if __name__ == "__main__":
    agente = Agente()
    avaliar(agente)
    pass
