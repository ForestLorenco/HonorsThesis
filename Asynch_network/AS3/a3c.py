import tensorflow as tf
import numpy as np
import math
import gym

import threading

import network
import a3c_thread

import const

import time


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


def _anneal_learning_rate(global_time_step):
    learning_rate = initial_learning_rate * (
            const.MAX_TIME_STEP - global_time_step) / const.MAX_TIME_STEP
    if learning_rate < 0.0:
        learning_rate = 0.0
    return learning_rate

ENV = gym.make(const.GAME)
ACTIONS = ENV.action_space.n

initial_learning_rate = log_uniform(const.INITIAL_ALPHA_LOW,
                                    const.INITIAL_ALPHA_HIGH,
                                    const.INITIAL_ALPHA_LOG_RATE)

current_lr = initial_learning_rate

global_network = network.Network(ACTIONS, -1)

global_t = 0

lr_input = tf.placeholder("float")

#Define stuff for optimization
#Action input for policy
pi = global_network.pi
a = tf.compat.v1.placeholder("float", [None, ACTIONS])

# temporary difference (R-V) (input for policy)
temp_diff = tf.compat.v1.placeholder("float", [None])

# avoid NaN with clipping when value in pi becomes zero
log_pi = tf.log(tf.clip_by_value(pi, 1e-20, 1.0))

# policy entropy
entropy = -tf.reduce_sum(pi * log_pi, reduction_indices=1)

# policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
policy_loss = - tf.reduce_sum(
    tf.reduce_sum(tf.multiply(log_pi, a), reduction_indices=1) * temp_diff + entropy * const.ENTROPY_BETA)

#input for the value function
r = tf.compat.v1.placeholder("float", [None])

# value loss (output)
# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
value_loss = 0.5 * tf.nn.l2_loss(r - global_network.v)

# gradienet of policy and value are summed up
total_loss = policy_loss + value_loss
opt_vars = [a, temp_diff, r, global_network.s]
train_global = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr_input, decay=const.RMSP_ALPHA, momentum=0.0, epsilon=const.RMSP_EPSILON).minimize(total_loss)

threads = []
#create a new thread
for i in range(8):
    thread = a3c_thread.A3C_Thread(i, const.MAX_TIME_STEP, ACTIONS)
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
    global opt_vars
    global global_network
    global a
    global temp_diff
    global r
    global initial_learning_rate
    global current_lr
    global lr_input
    thread = threads[index]

    #make the game
    lock.acquire()
    env = gym.make(const.GAME)
    obs = env.reset()
    lock.release()

    thread.get_init_obs(obs)
    while global_t < const.MAX_TIME_STEP:
        t, vars = thread.step(sess, global_t, global_network, lock, saver, env)

        current_lr = _anneal_learning_rate(global_t)

        global_t += t
        batch_si, batch_a, batch_td, batch_R = vars[0], vars[1], vars[2], vars[3]

        sess.run(train_global,
                 feed_dict={a: batch_a,
                            r: batch_R,
                            temp_diff: batch_td,
                            global_network.s: batch_si,
                            lr_input: current_lr
                            })



if __name__ == "__main__":
    train_threads = []
    lock = threading.Lock()
    for i in range(8):
        train_threads.append(threading.Thread(target=train, args=(i,lock)))
    start = time.time()

    # start all the threads
    for x in train_threads:
        x.start()

    # Join them all when they are done
    for x in train_threads:
        x.join()

    end = time.time()
    print("Model took {} time to train".format(end-start))