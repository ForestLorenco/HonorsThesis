import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.optimizers import RMSprop
from keras.initializers import VarianceScaling
from keras.layers import Flatten
from keras.models import load_model

import collections
import random
import numpy as np

import os.path

import time

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #things for the conv network
        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
        self.conv_strides = [4, 2, 1]
        self.input_height = 88
        self.input_width = 80
        self.input_channels = 1

        #things for the learning
        self.memory = collections.deque(maxlen=20000)
        self.gamma = 0.95  # this is discount rate
        self.start_epsilon = 1.0
        self.epsilon = self.start_epsilon# exploration rate
        self.epsilon_min = 0.1  # min for exploration rate
        #self.epsilon_decay = 0.995  # how fast epsilon decays
        self.lr = 0.001
        self.momentum = 0.95

        #build the model
        self.model = self._build_model()
        #builds the target model used to predict target Q values
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        first = True
        for n_maps, kernel_size, strides in zip(self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides):
            if first:
                #print("hello")
                model.add(Conv2D(input_shape=(self.input_height, self.input_width, self.input_channels), filters=n_maps, kernel_size=kernel_size,
                    strides=strides, padding="same", activation='relu', data_format="channels_last", kernel_initializer=VarianceScaling()))
                first = False
            else:

                model.add(Conv2D(filters=n_maps,
                                 kernel_size=kernel_size,
                                 strides=strides, padding="same", activation='relu',
                                 kernel_initializer=VarianceScaling()))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=VarianceScaling(), name='dense1'))
        model.add(Dense(self.action_size, activation='relu', kernel_initializer=VarianceScaling(), name='dens2'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.lr, momentum=self.momentum))
        return model

    def remember(self, action, state, reward, next_state, done):
        self.memory.append((action, state,reward,next_state,done))

    # gets the next action of the agent
    def act(self, state):
        # this decides to do a random action based off the epsilon rate
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #print("yup")
        state = np.reshape(state, [-1,88,80,1])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    #epsilon decay
    def epsilonDecay(self, t):
        if self.start_epsilon > self.epsilon_min:
            self.epsilon = t*(self.epsilon_min- self.start_epsilon)/(10**6) + self.start_epsilon

    # train the network off the memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)  # gets a sample of size batch_size from memory
        states = []
        target_fs = []
        rewards = []
        for state, action, reward, nextstate, done in minibatch:
            target = reward

            if not done:
                nextstate = np.reshape(nextstate, [-1, 88, 80, 1])
                #compute target state with the target model for higher consistancy
                target = (reward + self.gamma * np.amax(self.target_model.predict(nextstate)[0]))
            temp = np.reshape(state, [-1, 88, 80, 1])
            target_f = self.model.predict(temp)
            target_f[0][action] = target
            target_fs.append(target_f)

            states.append(state)
        states = np.array(states)
        target_fs = np.array(target_fs)
        target_fs = target_fs.reshape(32,9)
        #print(states.shape, target_fs.shape)
        start = time.time()
        self.model.predict(states)
        self.model.fit(states, target_fs, epochs=1, verbose=0)
        end = time.time()
        #print("Time for model.fit to run is", end-start) #better but still takes far too long

    def updateTarget(self):
        self.target_model.set_weights(self.model.get_weights())


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
    k = 4
    env = gym.make("MsPacman-v0")
    done = False  # env needs to be reset

    #variables for training
    skip_start = 90 #we are going to skip start of every game as it is just waiting
    batch_size = 32


    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = DQNAgent(state_space, action_space)
    if os.path.isfile("pacman_dqn_model2.h5"):
        print("Loading model")
        agent.model = load_model("pacman_dqn_model2.h5")
        agent.target_model = load_model("pacman_dqn_targemodel.h5")
    t = 0
    while t < 10**7:
        obs = env.reset()


        for skip in range(skip_start):  # skip the start of each game
            obs, reward, done, info = env.step(0)

        state = preprocess_observation(obs)
        #state = np.reshape(state, [-1, 88, 80, 1])

        total_reward = 0
        j = 0
        action = agent.act(state)
        while not done:
            j += 1
            t += 1
            #env.render()
            '''
            the agent sees and selects actions on every k
            th frame instead of every
            frame, and its last action is repeated on skipped frames
            Since running the emulator forward for one
            step requires much less computation than having the agent select an action, this technique allows
            the agent to play roughly k times more games without significantly increasing the runtime
            '''
            if j % k == 0:
                action = agent.act(state)

            next_obs, reward, done, _ = env.step(action)
            next_state = preprocess_observation(next_obs)
            #next_state = np.reshape(next_state, [-1, 88, 80, 1])
            #print(state.shape, next_state.shape)


            total_reward += reward
            reward = np.clip(reward, -1, 1)
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("score: {}, e: {:.2} frames:{}".format(total_reward, agent.epsilon, t))
                break
            #train agent with the experience of the episode

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if t % 10 == 0:
                agent.model.save('pacman_dqn_model2.h5')
                agent.target_model.save("pacman_dqn_targemodel.h5")

            if t%50 == 0:
                agent.updateTarget()

            agent.epsilonDecay(t)
