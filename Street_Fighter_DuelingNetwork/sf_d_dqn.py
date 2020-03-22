import cv2
import numpy as np
import random 
import time

import signal

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

import tensorflow as tf

import os.path
import collections
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
assert len(K.tensorflow_backend._get_available_gpus()) > 0
class SF_Dueling:

    def __init__(self, start_epsilon = 1.0, resume=0):
    #things for the learning
        self.env = SFENV(multi=False,skip=True)
        obs = self.env.reset()
        x_t = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        self.action_size = self.env.actions_space
        self.memory = collections.deque(maxlen=20000)
        self.gamma = 0.95  # this is discount rate
        self.start_epsilon = start_epsilon
        self.epsilon = self.start_epsilon# exploration rate
        self.epsilon_min = 0.01  # min for exploration rate
        self.decay_rate = 0.99
        self.lr = 0.0001

        self.total_time = 15*(10**5)
        self.scores = []
        self.stop_requested = False

        #build the model
        self._build_model()
        self.resume = resume

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
        state = np.reshape(state, (1,84,84,4))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    #epsilon decay
    def epsilonDecay(self, t):
        if self.epsilon > self.epsilon_min:
            self.epsilon = t*(self.epsilon_min- self.start_epsilon)/(self.total_time) + self.start_epsilon
    
    # train the network off the memory
    def replay(self, batch_size, t):
        minibatch = random.sample(self.memory, batch_size)  # gets a sample of size batch_size from memory
        s_batch = []
        targets = np.zeros((batch_size, self.action_size))
        i = 0
        for state, action, reward, nextstate, done in minibatch:
            s_batch.append(state)
            state = np.reshape(state, (1,84,84,4))
            nextstate = np.reshape(nextstate, (1,84,84,4))
            targets[i] = self.model.predict(state, batch_size = 1)
            fut_action = self.target_model.predict(nextstate, batch_size = 1)
            targets[i, action] = reward
            if done == False:
                targets[i, action] += self.decay_rate * np.max(fut_action)
            i+=1
        s_batch = np.array(s_batch)
        loss = self.model.train_on_batch(s_batch, targets)
        
        if t %100 == 0:
            print("Loss of model is", loss)

    
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

    def write_data(self):
        f = open("Data.csv", "w")
        f.write("Score\n")
        for d in self.scores:
            f.write(str(d)+"\n")
        f.close()

    def train(self):
        def signal_handler(signal, frame):
            print('You pressed Ctrl+C!')
            self.stop_requested = True

        if os.path.isfile("sf2_dqn_model2.h5"):
            print("Loading model")
            self.load_network("sf2_dqn_model2.h5")
        
        signal.signal(signal.SIGINT, signal_handler)

        
        
        t = self.resume
        total_reward = 0
        batch_size = 32
        start_time = time.time()
        while t < self.total_time:
            
            if self.stop_requested:
                self.save_network('sf2_dqn_model2.h5')
                self.write_data()
                exit(0)
            
            t += 1

            #Get the action
            #print(self.s_t.shape)
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
                print("### Score:{} Epsilon: {} Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                        total_reward, self.epsilon, t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
                total_reward = 0
            
            if len(self.memory) > batch_size:
                self.replay(batch_size, t)
            
            if t % 10000 == 0:
                self.save_network('sf2_dqn_model2.h5')
                
            self.epsilonDecay(t)
        self.write_data()
        signal.pause()
            
if __name__ == "__main__":
    
    print(tf.test.is_gpu_available()) # True/False

    # Or only check for gpu's with cuda support
    print(tf.test.is_gpu_available(cuda_only=True)) 

    agent = SF_Dueling(start_epsilon=0.6, resume = 1000000)
    agent.train()
