import tensorflow as tf
import gym
import time
import numpy as np
import random

import threading

class Network_Maker:
    def __init__(self, action_size):
        self.action_size = action_size

        # things for the conv network
        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8, 8, 4), (4, 4, 32), (3, 3, 64)]
        self.conv_strides = [4, 2, 1]
        self.input_height = 88
        self.input_width = 80
        self.input_channels = 1


    def weight_var(self, shape):
        """
        Creates a tensor that is shape with random numbers from a normal distritution with stdev 0.01
        :param shape:
        :return a tf variable of shape shape:
        """
        init = tf.random.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(init)

    def bias_var(self, shape):
        """Creates a tensor that is shape with values 0.01
        :param shape:
        :return a tf variable of shape shape:
        """
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init)

    def conv2d(self, input, W, stride):
        """
        Creates a conv2d tensor for a layer in a neural network
        :param input:
        :param W:
        :param stride:
        :return a conv2d nn tensor:
        """
        return tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        """
        Performs max_pool for the output of the convolutional layer
        :param x:
        :return a nn.max_pool tensor:
        """
        return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def createNetwork(self):
        """
        Creates and returns DQN network for agent
        :return output and input layer, as well as all weight and bias tensors for nn:
        """
        #input layer
        inp_layer = tf.compat.v1.placeholder("float", [None, self.input_height, self.input_width, self.input_channels])

        #first convolutional layer tensors
        W_conv1 = self.weight_var([8, 8, 1, 32])
        b_conv1 = self.bias_var([32])

        W_conv2 = self.weight_var([4, 4, 32, 64])
        b_conv2 = self.bias_var([64])

        W_conv3 = self.weight_var([3, 3, 64, 64])
        b_conv3 = self.bias_var([64])

        #Fully connected layer tensors
        W_fc = self.weight_var([256, 256])
        b_fc = self.bias_var([256])

        #ouptput layer tensors
        W_out = self.weight_var([256, self.action_size])
        b_out = self.bias_var([self.action_size])

        #Hidden layer declerations
        h_conv1 = tf.nn.relu(self.conv2d(inp_layer, W_conv1, self.conv_strides[0])+b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, self.conv_strides[1]) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, self.conv_strides[2]) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        #Flatten layer
        h_pool3 = tf.reshape(h_pool3, [-1,256])

        h_fc = tf.nn.relu(tf.matmul(h_pool3, W_fc)+b_fc)

        output = tf.matmul(h_fc, W_out)+b_out

        return inp_layer, output, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc, b_fc, W_out, b_out


def preprocess_observation(obs):
    """
    This function preprocesses image, normalizing color, grayscaling and improcing contrast
    :param obs:
    :return:
    """
    # We need to preprocess the images to speed up training
    main_char_color = np.array([86,138,89]).mean() #<- specific to mrs.pacman, will need to specialize this for each game
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==main_char_color] = 0 # Improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

def copy_target_network(sess):
    """
    Updates the weights of the target network with the weights of the online network
    :param sess:
    :return:
    """
    sess.run(copy_Otarget)

def train_agent(id, sess, lock):
    """

    :param id thread id:
    :param sess session for tensor:
    :param lock thread locks:
    :return:
    """

    #declare shared TMAX and T parameter so that they can be changed in function
    global TMAX, T

    #Create a new environment
    lock.acquire()
    env = gym.make(GAME)
    obs = env.reset()
    lock.release()

    #Here we initialize the network gradients
    in_batch = []
    a_batch = []
    y_batch = []

    """
    skip_start = 90
    for skip in range(skip_start):  # skip the start of each game
        lock.acquire()
        obs, reward, done, info = env.step(0)
        lock.release()
    """

    obs = preprocess_observation(obs)

    #Sleep the network to get everything to start
    time.sleep(3*id)

    #Initialize target network weights
    copy_target_network(sess)

    #Get a random epsilon for each thread
    e_index = random.randrange(EPSILONS)
    init_epsilon = INITIAL_EPSILONS[e_index]
    final_epsilon = FINAL_EPSILONS[e_index]
    epsilon = init_epsilon

    print("Thread {} starting with initial epsilon {} and final epsilon {}".format(id, init_epsilon, final_epsilon))

    #Thread step counter
    t = 0

    score = 0
    while T < TMAX:
        '''
        #If we have reset the game, skip startup frames
        if skip:
            skip = False
            for skip in range(skip_start):  # skip the start of each game
                lock.acquire()
                obs, reward, done, info = env.step(0)
                lock.release()
            obs = preprocess_observation(obs)
        '''

        #Choose action

        out_t = O_net_out.eval(session=sess, feed_dict={O_net_in:[obs]}) #Get action from online_nn
        a_t = np.zeros([ACTIONS])

        action_index = 0
        if random.random() <= epsilon: #might add observation period
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(out_t)
            a_t[action_index] = 1

        #Decay epsilon
        if epsilon > final_epsilon: #may add observation period
            epsilon -= (init_epsilon- final_epsilon)/EXPLORE

        #Run the  action and observe reward and state
        lock.acquire()
        obs1, reward, done, info = env.step(np.argmax(a_t))
        lock.release()

        score += reward

        #Normalize the obs
        obs1 = preprocess_observation(obs1)

        #Accumulate the gradients
        out_t1 = T_net_out.eval(session=sess, feed_dict={T_net_in:[obs1]})

        if done:
            y_batch.append(reward)
        else:
            y_batch.append(reward+GAMMA*np.max(out_t1))

        a_batch.append(a_t)
        in_batch.append(obs)

        #set current state to next state
        obs = obs1

        #Increment Global, and local variables
        T += 1
        t += 1

        if T % Itarget == 0:
            #update the target network with the online
            copy_target_network(sess)

        if t%Iasync == 0 or done:
            #If we are done or every Iasync steps
            if in_batch: #If we have values in in batch
                train_O.run(session=sess, feed_dict={y: y_batch,
                                                     a: a_batch,
                                                     O_net_in: in_batch})

            #Clear the gradients
            in_batch = []
            a_batch = []
            y_batch = []

        if t % 5000 == 0:
            saver.save(sess, 'asyn_network_saves/' + GAME + '-dqn', global_step=t)

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if done:
            skip = True
            env.reset()
            print("THREAD:{} | STATE:{} | REWARD:{} | TIME:{} | EPSILON: {}".format(id, state, score, t, epsilon))
            score = 0
"""
Decleration of global variables, and networks for threads
"""
GAME = "SpaceInvaders-v0"
ENV = gym.make(GAME)


TMAX = 5000000*10 #Max amount of training steps
EXPLORE = 400000*10 # Frames over which to anneal epsilon
OBSERVE = 0

T = 0 #Initial training steps
Itarget = 10000 #Num iterations before updating target network
Iasync = 5 #Num iterations before updating the online network

GAMMA = 0.99 # Decay rate of past observations
FINAL_EPSILONS = [0.1, 0.01, 0.05] # Final values of epsilon
INITIAL_EPSILONS = [0.4, 0.3, 0.3] # Starting values of epsilon
EPSILONS = 3
ACTIONS = ENV.action_space.n

#Declare the online network
O_net_in, O_net_out, Wo_conv1, bo_conv1, Wo_conv2, bo_conv2, Wo_conv3, bo_conv3, Wo_fc, bo_fc, Wo_out, bo_out = Network_Maker(ACTIONS).createNetwork()

#Declare training tensors and define the cost functions, and optimizer
a = tf.compat.v1.placeholder("float", [None, ACTIONS])
y = tf.compat.v1.placeholder("float", [None])
O_out_action = tf.reduce_sum(tf.multiply(O_net_out, a), reduction_indices=1)
cost_O = tf.reduce_mean(tf.square(y - O_out_action))
train_O = tf.compat.v1.train.RMSPropOptimizer(0.001, decay=0.99).minimize(cost_O) #using the rmsprop, with decay and momentum

#Create the target network
T_net_in, T_net_out, Wt_conv1, bt_conv1, Wt_conv2, bt_conv2, Wt_conv3, bt_conv3, Wt_fc, bt_fc, Wt_out, bt_out = Network_Maker(ACTIONS).createNetwork()
#Creates a tensor of copy operations
copy_Otarget = [Wt_conv1.assign(Wo_conv1), bt_conv1.assign(bo_conv1), Wt_conv2.assign(Wo_conv2),
                bt_conv2.assign(bo_conv2), Wt_conv3.assign(Wo_conv3), bt_conv3.assign(bo_conv3), Wt_fc.assign(Wo_fc),
                bt_fc.assign(bo_fc), Wt_out.assign(Wo_out), bt_out.assign(bo_out)]

#initialize session variables
sess = tf.compat.v1.InteractiveSession()

#initialize a saver to save the model
saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())

#Restore saved network
checkpoint = tf.train.get_checkpoint_state("asyn_network_saves")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Loaded Network: {}".format(checkpoint.model_checkpoint_path))

if __name__ == "__main__":
    #Start a timer
    start = time.time()

    #This is initializing a lock to prevent race conditions when writing to the networks
    lock = threading.Lock()

    #Now we create all the threads for the agents
    threads = []
    num_threads = 10

    #Append all threads to a list of threads that run the train agent model
    for i in range(num_threads):
        thread = threading.Thread(target=train_agent, args=(i, sess, lock))
        threads.append(thread)

    #start all the threads
    for x in threads:
        x.start()

    #Join them all when they are done
    for x in threads:
        x.join()

    #End the timer
    end = time.time()

    print("Finished training model. Took: {}hr {}min {}sec".format((end-start)//360, ((end-start)%360)//60, ((end-start)%360)%60))



