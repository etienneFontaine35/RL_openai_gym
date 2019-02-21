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


def spaceNumber(rawState) :
    x = rawState[0]
    x_dot = rawState[1]
    theta = rad2degree(rawState[2])
    theta_dot = rad2degree(rawState[3])

    if (x < -0.8) :
        state = 0
    elif (x < 0.8) :
        state = 1
    else :
        state = 2
    
    if (x_dot < -0.5) :
        state += 0;
    elif (x_dot < 0.5) :
        state += 3
    else :
        state += 6

    if (theta < -6) :
        state += 0
    elif (theta < -1) :
        state += 9
    elif (theta < -0) :
        state += 18
    elif (theta < 1) :
        state += 27
    elif (theta < 6) :
        state += 36
    else :
        state += 45

    if (theta_dot < -50) :
        state += 0
    elif (theta_dot < 50) :
        state += 54
    else :
        state += 108
    
    return state


# initialisation des entités
cartEnv = gym.make('CartPole-v1')

MIN_STATE = cartEnv.observation_space.low
MAX_STATE = cartEnv.observation_space.high
print("Lower bounderies : {}".format(MIN_STATE))
print("Upper bounderies : {}".format(MAX_STATE))

NUM_ACTIONS = cartEnv.action_space.n

sys.exit(42)
for episode in range(1) :
    cartEnv.reset()
    done = False
    while done != True :
        newState, reward, done, info = cartEnv.step(cartEnv.action_space.sample())
        print("Etat : {}\nReward : {}\n".format(newState, reward))
        # cartEnv.render()
    print("Episode {}".format(episode))
    