import gym
import random
import numpy as np
import collections


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95 #this is discount rate
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.01 #min for exploration rate
        self.epsilon_decay = 0.995 #how fast epsilon decays
        self.lr = 0.001
        self.model = self._build_model()

    #build a model with 3 layers, finally with a adam optimizer
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    #remember previous things from the model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action, reward, next_state, done))

    #gets the next action of the agent
    def act(self, state):
        #this decides to do a random action based off the epsilon rate
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    #train the network off the memory
    def replay(self,batch_size):
        minibatch = random.sample(self.memory, batch_size) #gets a sample of size batch_size from memory
        for state, action, reward, nextstate, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(nextstate)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            start = time.time()
            self.model.fit(state, target_f, epochs=1, verbose=0)
            end = time.time()
            print("Time for model.fit to run is", end - start)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False

    episodes = 250
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time_t in range(500):
            env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1,state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e,episodes,time_t, agent.epsilon))
                break
            #train agent with the experience of the episode
            if len(agent.memory) > 32:
                agent.replay(32)