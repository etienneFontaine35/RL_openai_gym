import gym
import numpy as np
from time import sleep
import math
import sys
import os
import matplotlib.pyplot as plt


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


def simulate(alpha_param=None, epsilon_param=None, gamma_param=0.999, buckets_param=(1, 1, 6, 12,)) :
    # initialisation des entités
    cartEnv = gym.make('CartPole-v1')

    buckets = buckets_param
    NUM_STATES = buckets[0] * buckets[1] * buckets[2] * buckets[3]
    Qmatrix = np.zeros(buckets + (cartEnv.action_space.n,))
    gamma = gamma_param # discount factor, what importance we give to future rewards
    minAlpha = 0.1 # minimal learning rate, what importance we give to the new obtained values
    minEpsilon = 0.1 # minimal probability to choose a random action (exploration) instead of taking a rewarding one (exploitation)
    alpha = alpha_param
    epsilon = epsilon_param
    dynamicAlpha = (alpha == None)
    dynamicEpsilon = (epsilon == None)
    print("Continuous space discretized into {} states {}".format(NUM_STATES, buckets))
    print("Parameters : alpha = {}, gamma = {}, epsilon = {}".format(alpha, gamma, epsilon))

    MAX_EPISODES = 5000

    DISPLAY = False
    FREQ_LOG = -1

    episode = 0
    consecutiveSuccess = 0

    while True :
        done = False
        totalReward = 0.
        state = cartEnv.reset()
        numTimeSteps = 0

        if dynamicEpsilon :
            epsilon = epsilonDecay(episode, minEpsilon)

        if dynamicAlpha : 
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

        if FREQ_LOG != -1 and episode % FREQ_LOG == 0 :
            print("Episode {} : up for {} timesteps".format(episode, numTimeSteps))

        episode += 1
        
        if numTimeSteps > 195 :
            consecutiveSuccess += 1
        else : 
            consecutiveSuccess = 0

        if consecutiveSuccess > 100:
            print("==> Problem solved after {} episodes !\n".format(episode))
            break
        
        if episode > MAX_EPISODES - 1 :
            print("=/=> Problem not solved, maximum number of episodes reached\n")
            episode = -1
            break

    return episode


if __name__ == "__main__" :
    buckets = (1, 1, 6, 12,)
    alpha_list = np.linspace(0.1, 0.9, 9)
    epsilon_list = np.linspace(0.1, 0.9, 9)
    gamma_list = np.linspace(0.93, 1., 50)
    
    gamma_resultats = np.zeros(len(gamma_list))
    gamma_probability_success = np.ones(len(gamma_list))

    for i, value in enumerate(gamma_list) :
        average_result = 0
        probability = 1.
        weight = 0
        print("Calcul de {}".format(i))
        for j in range(10) :
            tmp = simulate(gamma_param=value)
            if tmp != -1 :
                weight += 1.
                average_result += tmp
            else :    
                probability -= 0.1

        weight = max(1, weight)
        average_result /= weight
        print("Prob = {}, Result = {}\n\n".format(probability, average_result))
        gamma_probability_success[i] = probability
        gamma_resultats[i] = average_result

    print(gamma_resultats)
    print("\n")
    print(gamma_probability_success)

    plt.plot(gamma_list, gamma_resultats, 'b-')
    plt.ylabel('Average number of episodes before success')
    plt.xlabel('Gamma')
    plt.title('Influence of Gamma on learning speed (with alpha and epsilon decay)')
    plt.show()

    plt.plot(gamma_list, gamma_probability_success, 'r-')
    plt.ylabel('Success rate')
    plt.xlabel('Gamma')
    plt.title('Probability to achieve the task according to Gamma (over 10 trials)')
    plt.show()


    