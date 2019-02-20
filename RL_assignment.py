import gym
import numpy as np
from time import sleep
import sys
import os

def bestActionValue(env, state, valueArray, discount) : # value iteration
    bestAction = None
    bestValue = float('-inf')

    for action in range(0, env.action_space.n) :
        env.env.s = state
        newState, reward, done, info = env.step(action)
        value = reward + discount * valueArray[newState]
        if value > bestValue :
            bestValue = value
            bestAction = action
    
    return bestAction


# initialisation des entités
taxiEnv = gym.make('Taxi-v2')
taxiEnv.reset()

NUM_ACTIONS = taxiEnv.action_space.n
NUM_STATES = taxiEnv.observation_space.n
V = np.zeros(NUM_STATES, dtype=float)
Pi = np.zeros(NUM_STATES, dtype=int)
gamma = 0.9 # discount factor, quelle importance on attribue aux décisions futures
epsilon = 0.01 # seuil traduisant la stagnation de l'apprentissage


# Apprentissage

print("Nbre d'actions disponibles : {}".format(NUM_ACTIONS))
print("Nbre d'états possibles de l'environnement : {}".format(NUM_STATES))

iteration = 0

while True :

    biggestChange = 0
    
    for state in range(NUM_STATES) :
        oldValue = V[state]
        action = bestActionValue(taxiEnv, state, V, gamma)
        taxiEnv.env.s = state
        newState, reward, done, info = taxiEnv.step(action)
        V[state] = reward + gamma * V[newState]
        Pi[state] = action
        biggestChange = max(biggestChange, np.abs(oldValue - V[state]))
    
    iteration += 1

    if biggestChange < epsilon :
        print("Optimal policy found after {} iterations".format(iteration))
        break




# Test réel
totalReward = 0.
state = taxiEnv.reset()
os.system('clear')
taxiEnv.render()
done = False

while done != True :
    sleep(1)
    os.system('clear')
    action = Pi[state]
    state, reward, done, info = taxiEnv.step(action)
    totalReward += reward
    taxiEnv.render()

print("Reward : {}".format(totalReward))