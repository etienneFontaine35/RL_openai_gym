import gym
import numpy as np
from time import sleep
import sys
import os

# initialisation des entités
taxiEnv = gym.make('Taxi-v2')

NUM_ACTIONS = taxiEnv.action_space.n
NUM_STATES = taxiEnv.observation_space.n
Qmatrix = np.zeros([NUM_STATES, NUM_ACTIONS])
gamma = 0.9 # discount factor, quelle importance on attribue aux décisions futures
alpha = 0.9 # learning rate, quelle importance on attribue à la valeur apprise
epsilon = 0.01 # seuil traduisant la stagnation de l'apprentissage

FRAME_DELAY = 0.04
EPISODE_TRANSITION_DELAY = 1.
DISPLAY = False

rapport = "Log de l apprentissage :\n"

for episode in range(1000) :
    done = False
    totalReward = 0.
    state = taxiEnv.reset()

    while done != True :
        if DISPLAY :
            sleep(FRAME_DELAY)
            os.system('clear')
            taxiEnv.render()
            print("Episode {}".format(episode))
        action = np.argmax(Qmatrix[state])
        newState, reward, done, info = taxiEnv.step(action)
        Qmatrix[state, action] += alpha * (reward + gamma * np.max(Qmatrix[newState]) - Qmatrix[state, action])
        totalReward += reward
        state = newState
    
    if DISPLAY :
        sleep(FRAME_DELAY)
        os.system('clear')
        taxiEnv.render()
        print("Episode {} --> end\nTotal reward : {}".format(episode, totalReward))
        sleep(EPISODE_TRANSITION_DELAY)

    if episode % 50 == 0 :
        rapport += "Episode {} : total reward of {}\n".format(episode, totalReward)

print(rapport)

