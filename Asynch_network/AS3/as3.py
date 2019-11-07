import tensorflow as tf
import numpy as np
import math
import gym

import threading

import network
import a3c_thread

from rmsprop_applier import RMSPropApplier

import const

import time


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

ENV = gym.make(const.GAME)
ACTIONS = ENV.action_space.n

initial_learning_rate = log_uniform(const.INITIAL_ALPHA_LOW,
                                    const.INITIAL_ALPHA_HIGH,
                                    const.INITIAL_ALPHA_LOG_RATE)

global_network = network.Network(ACTIONS, -1)

global_t = 0

lr_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = lr_input,
                              decay = const.RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = const.RMSP_EPSILON,
                              clip_norm = const.GRAD_NORM_CLIP)

threads = []
#create a new thread
for i in range(8):
    thread = a3c_thread.A3C_Thread(i, global_network, initial_learning_rate, grad_applier, const)
    threads.append(thread)

sess = tf.compat.v1.InteractiveSession()

#initialize a saver to save the model
saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())

#Restore saved network
checkpoint = tf.train.get_checkpoint_state("a3c_saves")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Loaded Network: {}".format(checkpoint.model_checkpoint_path))

def train(index, lock):
    global global_t
    thread = threads[index]

    while global_t < const.MAX_TIME_STEP:
        t = thread.step(sess, global_t, global_network, lock, saver)

        global_t += t

if __name__ == "__main__":
    train_threads = []
    lock = threading.Lock()
    for i in range(8):
        train_threads.append(threading.Thread(target=train, args=(i,lock)))
    start = time.time()

    # start all the threads
    for x in threads:
        x.start()

    # Join them all when they are done
    for x in threads:
        x.join()

    end = time.time()
    print("Model took {} time to train".format(end-start))