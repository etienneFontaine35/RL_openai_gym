import gym
import numpy as np
from time import sleep
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


# initialisation des entités
cartEnv = gym.make('CartPole-v1')

xBuckets = np.array([])
xDotBuckets = np.array([]) # np.array([-0.5, 0.5])
thetaBucket = np.array([deg2rad(-6), deg2rad(-1), deg2rad(0), deg2rad(1), deg2rad(6)])
thetaDotBucket = np.array([deg2rad(-50), deg2rad(50)])

NUM_STATES = (xBuckets.size+1) * (xDotBuckets.size + 1) * (thetaBucket.size + 1) * (thetaDotBucket.size + 1)
print("The continuous space is discretized into {} discretized states\n".format(NUM_STATES))
NUM_ACTIONS = cartEnv.action_space.n
Qmatrix = np.zeros([NUM_STATES, NUM_ACTIONS])
gamma = 1. # discount factor, what importance we give to future rewards
alpha = 0.7 # learning rate, what importance we give to the new obtained values
epsilon = 0.3 # probability to choose a random action (exploration) instead of taking a rewarding one (exploitation)

DISPLAY = False
FREQ_LOG = 20


for episode in range(1000) :
    done = False
    totalReward = 0.
    state = cartEnv.reset()
    numTimeSteps = 0

    while done != True :
        stateNum = stateNumber(state, xBuckets, xDotBuckets, thetaBucket, thetaDotBucket)
        action = epsGreedyPolicy(cartEnv, Qmatrix, epsilon, stateNum)
        
        if DISPLAY :
            cartEnv.render()

        newState, reward, done, info = cartEnv.step(action)
        newStateNum = stateNumber(newState, xBuckets, xDotBuckets, thetaBucket, thetaDotBucket)
        # print("State {}".format(newStateNum))

        Qmatrix[stateNum, action] += alpha * (reward + gamma * np.max(Qmatrix[newStateNum]) - Qmatrix[stateNum, action])
        totalReward += reward
        numTimeSteps += 1
        state = newState
    if (episode % FREQ_LOG == 0) :
        print("Episode {} : up for {} timesteps".format(episode, numTimeSteps))

    