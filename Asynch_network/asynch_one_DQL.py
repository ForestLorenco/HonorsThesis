import tensorflow as tf
import gym
import time
import numpy as np
import random

import threading
import cv2

import networkmaker

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

    x_t = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    aux_s = s_t

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

        out_t = O_net_out.eval(session=sess, feed_dict={O_net_in:[s_t]}) #Get action from online_nn
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

        x_t1 = cv2.cvtColor(cv2.resize(obs1, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis=2)
        s_t1 = np.append(aux_s, x_t1, axis=2)

        score += reward

        #Accumulate the gradients
        out_t1 = T_net_out.eval(session=sess, feed_dict={T_net_in:[s_t1]})

        if done:
            reward = -10
            y_batch.append(reward)
        else:
            y_batch.append(reward+GAMMA*np.max(out_t1))

        a_batch.append(a_t)
        in_batch.append(s_t)

        #set current state to next state
        s_t = s_t1

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
Itarget = 20000 #Num iterations before updating target network
Iasync = 5 #Num iterations before updating the online network

GAMMA = 0.99 # Decay rate of past observations
FINAL_EPSILONS = [0.1, 0.01, 0.05] # Final values of epsilon
INITIAL_EPSILONS = [0.4, 0.3, 0.3] # Starting values of epsilon
EPSILONS = 3
ACTIONS = ENV.action_space.n

#Declare the online network
O_net_in, O_net_out, Wo_conv1, bo_conv1, Wo_conv2, bo_conv2, Wo_conv3, bo_conv3, Wo_fc, bo_fc, Wo_out, bo_out = networkmaker.Networkmaker(ACTIONS).createNetwork()

#Declare training tensors and define the cost functions, and optimizer
a = tf.compat.v1.placeholder("float", [None, ACTIONS])
y = tf.compat.v1.placeholder("float", [None])
O_out_action = tf.reduce_sum(tf.multiply(O_net_out, a), reduction_indices=1)
cost_O = tf.reduce_mean(tf.square(y - O_out_action))
train_O = tf.compat.v1.train.RMSPropOptimizer(0.001, decay=0.99).minimize(cost_O) #using the rmsprop, with decay and momentum

#Create the target network
T_net_in, T_net_out, Wt_conv1, bt_conv1, Wt_conv2, bt_conv2, Wt_conv3, bt_conv3, Wt_fc, bt_fc, Wt_out, bt_out = networkmaker.Networkmaker(ACTIONS).createNetwork()
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



