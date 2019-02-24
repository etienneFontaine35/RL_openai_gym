import gym
import numpy as np
from time import sleep
import math
import sys
import os


# TODO: maintenant qu'on a la discrétisation de l'espace, il suffit juste de réappliquer la même
#       méthode que pour le taxi, en rajoutant le eps-greedy algorithme ()

# TODO: levier de variation :
#           - eps-greedy, eps decaying greedy,
#           - la taille de buckets
#           - le nombre de variables que l'on considère
#           - la valeur des paramètres : alpha, gamma
#           

def rad2degree(angle) :
    return angle * 180 / np.pi


def deg2rad(angle) :
    return angle * np.pi / 180


def discretize(state, env, buckets):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)


def epsGreedyPolicy(env, Qmatrix, eps, stateNum) :
    if np.random.rand(1) < eps : # exploration
        action = env.action_space.sample()
    else : # exploitation
        action = np.argmax(Qmatrix[stateNum])
    
    return action


def epsilonDecay(episode, minEps):
    return max(minEps, min(1, 1.0 - np.log10((episode+1)/25.0)))


def alphaDecay(episode, minAlpha):
    return max(minAlpha, min(0.5, 1.0 - np.log10((episode+1)/25.0)))


# initialisation des entités
cartEnv = gym.make('CartPole-v1')

buckets = (1, 1, 6, 12,)
Qmatrix = np.zeros(buckets + (cartEnv.action_space.n,))
gamma = 0.999 # discount factor, what importance we give to future rewards
minAlpha = 0.1 # minimal learning rate, what importance we give to the new obtained values
minEpsilon = 0.1 # minimal probability to choose a random action (exploration) instead of taking a rewarding one (exploitation)

MAX_EPISODES = 1000

DISPLAY = False
FREQ_LOG = 1 # MAX_EPISODES // 10 + 1

episode = 0

while True :
    done = False
    totalReward = 0.
    state = cartEnv.reset()
    numTimeSteps = 0
    epsilon = epsilonDecay(episode, minEpsilon)
    alpha = alphaDecay(episode, minAlpha)

    while done != True :

        stateNum = discretize(state, cartEnv, buckets)
        action = epsGreedyPolicy(cartEnv, Qmatrix, epsilon, stateNum)
        
        if DISPLAY :
            cartEnv.render()

        newState, reward, done, info = cartEnv.step(action)
        newStateNum = discretize(newState, cartEnv, buckets)

        Qmatrix[stateNum][action] += alpha * (reward + gamma * np.max(Qmatrix[newStateNum]) - Qmatrix[stateNum][action])
        totalReward += reward
        numTimeSteps += 1
        state = newState

    if (episode % FREQ_LOG == 0) :
        print("Episode {} : up for {} timesteps".format(episode, numTimeSteps))
    
    episode += 1

    if episode > MAX_EPISODES - 1 :
        break

    