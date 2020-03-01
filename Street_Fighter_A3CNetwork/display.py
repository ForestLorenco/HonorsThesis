# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import cv2

from sfenv import SFENV
from network import GameACFFNetwork, GameACLSTMNetwork 
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM

def choose_action(pi_values):
  return np.random.choice(range(len(pi_values)), p=pi_values)  

# use CPU for display tool
device = "/cpu:0"

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, -1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")

env = SFENV(render=True, multi=False)
obs = env.reset()
x_t = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

while True:
  pi_values = global_network.run_policy(sess, s_t)

  action = choose_action(pi_values)
  obs, reward, terminal, info = env.step(action)

  if terminal:
    env.close()
  else:
    x_t1 = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84))
    x_t1 = np.reshape(x_t1, (84, 84, 1))
    aux_s = np.delete(s_t, 0, axis=2)
    s_t = np.append(aux_s, x_t1, axis=2)
    