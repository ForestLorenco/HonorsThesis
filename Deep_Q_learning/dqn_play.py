import gym

from keras.models import load_model


import numpy as np

import os.path

import matplotlib.pyplot as plt

from ms_pacman_dqn2 import DQNAgent

#This function preprocesses the image
def preprocess_observation(obs):
    # We need to preprocess the images to speed up training
    mspacman_color = np.array([210, 164, 74]).mean()
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

if __name__ == '__main__':
    env = gym.make("MsPacman-v0")
    done = False  # env needs to be reset

    #variables for training
    skip_start = 90 #we are going to skip start of every game as it is just waiting
    batch_size = 32


    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = DQNAgent(state_space, action_space)
    agent.epsilon = agent.epsilon_min
    if os.path.isfile("pacman_dqn_model2.h5"):
        print("Loading model")
        agent.model = load_model("pacman_dqn_model2.h5")
        agent.target_model = load_model("pacman_dqn_targemodel.h5")
    t = 0
    rewards = []
    episodes = 100
    for i in range(episodes):
        obs = env.reset()
        if i%10 == 0:
            env.render()

        for skip in range(skip_start):  # skip the start of each game
            if i % 10 == 0:
                env.render()
            obs, reward, done, info = env.step(0)

        state = preprocess_observation(obs)
        #state = np.reshape(state, [-1, 88, 80, 1])

        total_reward = 0
        while not done:
            if i % 10 == 0:
                env.render()
            t += 1
            #env.render()
            action = agent.act(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = preprocess_observation(next_obs)
            #next_state = np.reshape(next_state, [-1, 88, 80, 1])
            #print(state.shape, next_state.shape)

            reward = reward
            total_reward += reward

            state = next_state

            if done:
                rewards.append(total_reward)
                print("episode: {}/{}, score: {}".format(i,episodes,total_reward))
                break
    plt.plot(rewards)
    plt.title('Pacman Score Over 100 Episodes')
    plt.ylabel('Scofe')
    plt.xlabel('Episode')
    plt.savefig('pacman_scores1.png')
    plt.show()