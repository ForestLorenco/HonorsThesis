import cv2
import numpy as np
import random 
import time

from sfenv import SFENV

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.optimizers import RMSprop
from keras.initializers import VarianceScaling
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Input
from keras.layers import merge
from keras.models import Model
from keras.layers import Lambda


import os.path
import collections

class SF_Dueling:

    def __init__(self):
    #things for the learning
        self.env = SFENV(multi=False)
        obs = self.env.reset()
        x_t = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        self.action_size = self.env.actions_space
        self.memory = collections.deque(maxlen=20000)
        self.gamma = 0.95  # this is discount rate
        self.start_epsilon = 1.0
        self.epsilon = self.start_epsilon# exploration rate
        self.epsilon_min = 0.01  # min for exploration rate
        #self.epsilon_decay = 0.995  # how fast epsilon decays
        self.lr = 0.0001

        self.total_time = 10**7
        self.scores = []

        #build the model
        self._build_model

    def _build_model(self):
        self.model = Sequential()
        input_layer = Input(shape = (84, 84, 4))
        conv1 = Conv2D(32, 8, 8, subsample=(4, 4), activation='relu')(input_layer)
        conv2 = Conv2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, 3, 3, activation = 'relu')(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        advantage = Dense(self.action_size)(fc1)
        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.action_size,))([advantage, value])

        self.model = Model(input=[input_layer], output=[policy])
        #online
        self.model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))

        #target
        self.target_model = Model(input=[input_layer], output=[policy])
        self.target_model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
        print("Successfully constructed networks.")
    
    def remember(self, action, state, reward, next_state, done):
        self.memory.append((action, state, reward,next_state,done))
    
    # gets the next action of the agent
    def act(self, state):
        # this decides to do a random action based off the epsilon rate
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #print("yup")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    #epsilon decay
    def epsilonDecay(self, t):
        if self.epsilon > self.epsilon_min:
            self.epsilon = t*(self.epsilon_min- self.start_epsilon)/(10**7) + self.start_epsilon
    
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
        self.model.predict(states)
        self.model.fit(states, target_fs, epochs=1, verbose=0)
        #print("Time for model.fit to run is", end-start) #better but still takes far too long
    
    def updateTarget(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Succesfully loaded network.")

    def train(self):
        def signal_handler(signal, frame):
            global stop_requested
            print('You pressed Ctrl+C!')
            stop_requested = True

        if os.path.isfile("pacman_dqn_model2.h5"):
            print("Loading model")
            self.load_network("pacman_dqn_model2.h5")
        
        t = 0
        total_reward = 0
        batch_size = 32
        start_time = time.time
        while t < self.total_time:
            
            if stop_requested:
                self.save_network('pacman_dqn_model2.h5')
            
            t += 1

            #Get the action
            action = self.act(self.s_t)

            prev = self.s_t
            obs, reward, terminal, info = self.env.step(action)
            x_t1 = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
            x_t1 = np.reshape(x_t1, (84, 84, 1))
            aux_s = np.delete(self.s_t, 0, axis=2)
            self.s_t = np.append(aux_s, x_t1, axis=2)

            total_reward += reward
            reward = np.clip(reward, -1, 1)
            self.remember(prev, action, reward, self.s_t, terminal)

            if terminal:
                #save score for graphing
                self.scores.append(total_reward)

                #reset env
                obs = self.env.reset()
                x_t = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
                x_t1 = np.reshape(x_t1, (84, 84, 1))
                self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

                #report performance
                elapsed_time = time.time() - start_time
                steps_per_sec = t / elapsed_time
                print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                        t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
            
            if len(self.memory) > batch_size:
                self.replay(batch_size)
            
            if t % 10000 == 0:
                self.save_network('pacman_dqn_model2.h5')
                
            self.epsilonDecay(t)
            
if __name__ == "__main__":
    agent = SF_Dueling()
    agent.train()
