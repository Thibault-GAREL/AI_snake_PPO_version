import progressbar
import time
import os

import snake
import exw
import compteur
import ia

# Au début de ton programme
executions = compteur.compter_executions()
print(f"Exécution n°{executions}")

# Créer l'agent PPO
state_dim = 16  # 8 distances aux obstacles + 8 distances à la nourriture
action_dim = 4  # UP, RIGHT, DOWN, LEFT
agent = ia.create_agent(state_dim, action_dim)

score_mean = []
score_temp = 0

modulo = (ia.nb_loop_train-1) // 100

fichier, wb, ws = exw.create("donnees2", "entrainement" + str(executions), "X", "Y")

model_name = "models_ppo" + str(executions)

if os.path.exists(model_name + "/snake_ppo_model.pth"):
    agent.load_model(model_name + "/snake_ppo_model.pth")
    print("Model loaded : " + model_name + "/snake_ppo_model.pth")
else:
    os.makedirs(model_name, exist_ok=True)
    print("Model created")



for episode in progressbar.progressbar(range(ia.nb_loop_train)):
    # state = [250, 353.5533905932738, 500, 424.26406871192853, 300, 353.5533905932738, 250, 353.5533905932738, 0, 0, 200, 0, 0, 0, 0, 0]

    score_temp += snake.game_loop(snake.rect_width, snake.rect_height, snake.display, agent)
    # print(f'longeur du buffer : {len(agent.replay_buffer.buffer)}')
    agent.train_step(batch_size=64)



    # Mise à jour du réseau cible toutes les N itérations
    if episode % modulo == 0 and episode != 0:
        score_mean.append(score_temp/modulo)
        agent.update_target()

        exw.ajouter_donnee(fichier, wb, ws, episode, score_temp/modulo, "Graphe de l'évolution des scores", "Episode", "Score")

        # Sauvegarder le modèle
        agent.save_model(model_name + '/snake_ppo_model.pth')

        score_temp = 0



# print (score_mean)