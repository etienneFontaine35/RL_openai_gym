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


def stateNumber(state, xBuckets, xDotBuckets, thetaBucket, thetaDotBucket) :
    stateNum = 0
    numStates = (xBuckets.size+1) * (xDotBuckets.size + 1) * (thetaBucket.size + 1) * (thetaDotBucket.size + 1)

    for i, buck in enumerate([xBuckets, xDotBuckets, thetaBucket, thetaDotBucket]) :
        numStates /= buck.size + 1
        rank = 0
        while (rank < buck.size) and (state[i] > buck[rank]) :
            rank += 1
        stateNum += rank*numStates

    return int(stateNum)


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

xBuckets = np.array([-0.8, 0.8])
xDotBuckets = np.array([-0.5, 0.5])
thetaBucket = np.array([deg2rad(-6), deg2rad(-1), deg2rad(0), deg2rad(1), deg2rad(6)])
thetaDotBucket = np.array([deg2rad(-25), deg2rad(25)])

NUM_STATES = (xBuckets.size+1) * (xDotBuckets.size + 1) * (thetaBucket.size + 1) * (thetaDotBucket.size + 1)
print("The continuous space is discretized into {} discretized states\n".format(NUM_STATES))
NUM_ACTIONS = cartEnv.action_space.n
# Qmatrix = np.zeros([NUM_STATES, NUM_ACTIONS])
# CECI est un test
Qmatrix = np.zeros(buckets + (cartEnv.action_space.n,))
gamma = 0.999 # discount factor, what importance we give to future rewards
minAlpha = 0.1 # minimal learning rate, what importance we give to the new obtained values
minEpsilon = 0.1 # minimal probability to choose a random action (exploration) instead of taking a rewarding one (exploitation)

MAX_EPISODES = 100

DISPLAY = False
FREQ_LOG = 1 # MAX_EPISODES // 10 + 1

episode = 0

# for episode in range(1000) :
while True :
    done = False
    totalReward = 0.
    state = cartEnv.reset()
    numTimeSteps = 0
    epsilon = epsilonDecay(episode, minEpsilon)
    alpha = alphaDecay(episode, minAlpha)

    while done != True :

        # stateNum = stateNumber(state, xBuckets, xDotBuckets, thetaBucket, thetaDotBucket)
        stateNum = getBox(state)
        action = epsGreedyPolicy(cartEnv, Qmatrix, epsilon, stateNum)
        
        if DISPLAY :
            cartEnv.render()

        newState, reward, done, info = cartEnv.step(action)
        # newStateNum = stateNumber(newState, xBuckets, xDotBuckets, thetaBucket, thetaDotBucket)
        newStateNum = getBox(newState)

        Qmatrix[stateNum, action] += alpha * (reward + gamma * np.max(Qmatrix[newStateNum]) - Qmatrix[stateNum, action])
        totalReward += reward
        numTimeSteps += 1
        state = newState

    # print(Qmatrix)
    # sleep(1)

    if (episode % FREQ_LOG == 0) :
        print("Episode {} : up for {} timesteps".format(episode, numTimeSteps))
    
    episode += 1

    if numTimeSteps > 195 or episode > MAX_EPISODES - 1 :
        break

    