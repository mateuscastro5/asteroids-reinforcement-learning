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
    total_recompensa = 0.0

    for _ in range(NUM_EPISODIOS):
        
        estado, info = env.reset()
        agente.reiniciar()
        truncado = False
        terminado = False
        passos = 0
        pontos = 0.0


        while not (terminado or truncado) and passos < MAX_PASSOS:
            acao = agente.agir(estado)
            estado, recompensa, terminado, truncado, info = env.step(acao)
            pontos += recompensa
            passos += 1

        total_recompensa += pontos

    env.close()
    return total_recompensa / NUM_EPISODIOS


def criar_populacao():

    lista = []

    for _ in range(TAMANHO_POPULACAO):
        init_agente = Agente()
        lista.append(init_agente)

    return lista
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
    pop = criar_populacao()
    print(len(pop))
    pass
