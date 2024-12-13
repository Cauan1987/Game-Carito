import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodios, em_treinamento=True, renderizar=False):
    env = gym.make('MountainCar-v0', render_mode='human' if renderizar else None)

    # Aumentando a discretização da posição e velocidade
    espaco_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 40)
    espaco_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 40)

    if em_treinamento:
        q = np.zeros((len(espaco_pos), len(espaco_vel), env.action_space.n))  # Cria uma tabela Q maior
    else:
        if os.path.exists('mountain_car.pkl'):  # Verifica se o arquivo existe
            with open('mountain_car.pkl', 'rb') as f:
                q = pickle.load(f)
        else:
            print("Arquivo 'mountain_car.pkl' não encontrado. Iniciando treinamento...")
            run(5000, em_treinamento=True, renderizar=False)  # Inicia o treinamento se o arquivo não existir
            return  # Após o treinamento, o código termina e o arquivo estará disponível

    taxa_aprendizado = 0.1
    fator_desconto = 0.99

    epsilon = 1.0
    taxa_de_diminuição_epsilon = 0.001
    rng = np.random.default_rng()

    recompensas_por_episodio = np.zeros(episodios)

    print("Iniciando o treinamento...")

    for i in range(episodios):
        estado = env.reset()[0]
        estado_pos = np.digitize(estado[0], espaco_pos)
        estado_vel = np.digitize(estado[1], espaco_vel)

        terminado = False
        recompensas = 0

        while not terminado and recompensas > -1000:
            if em_treinamento and rng.random() < epsilon:
                acao = env.action_space.sample()
            else:
                acao = np.argmax(q[estado_pos, estado_vel, :])

            novo_estado, recompensa, terminado, _, _ = env.step(acao)
            novo_estado_pos = np.digitize(novo_estado[0], espaco_pos)
            novo_estado_vel = np.digitize(novo_estado[1], espaco_vel)

            if em_treinamento:
                q[estado_pos, estado_vel, acao] = q[estado_pos, estado_vel, acao] + taxa_aprendizado * (
                    recompensa + fator_desconto * np.max(q[novo_estado_pos, novo_estado_vel, :]) - q[estado_pos, estado_vel, acao]
                )

            estado = novo_estado
            estado_pos = novo_estado_pos
            estado_vel = novo_estado_vel
            recompensas += recompensa

        epsilon = max(epsilon - taxa_de_diminuição_epsilon, 0.01)

        recompensas_por_episodio[i] = recompensas

    print("Treinamento concluído.")

    env.close()

    # Salvar a tabela Q após o treinamento
    if em_treinamento:
        try:
            with open('mountain_car.pkl', 'wb') as f:
                pickle.dump(q, f)
            print("Arquivo 'mountain_car.pkl' salvo com sucesso.")
        except Exception as e:
            print(f"Erro ao salvar o arquivo: {e}")

    # Média das recompensas
    recompensas_media = np.zeros(episodios)
    for t in range(episodios):
        recompensas_media[t] = np.mean(recompensas_por_episodio[max(0, t-100):(t+1)])
    plt.plot(recompensas_media)
    plt.xlabel('Episódios')
    plt.ylabel('Recompensas Médias')
    plt.title('Desempenho do MountainCar com Q-Learning')
    plt.savefig(f'mountain_car.png')

if __name__ == '__main__':
    #run(180, em_treinamento=True, renderizar=True)  # Treinamento
    run(180, em_treinamento=False, renderizar=True)  # Teste após treinamento
