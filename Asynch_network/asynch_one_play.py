import tensorflow as tf
import gym
import cv2
import numpy as np

import networkmaker

def play_game():
    env = gym.make(GAME)

    obs = env.reset()
    obs = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((obs, obs, obs, obs), axis=2)
    aux_s = s_t

    done = False
    while not done:
        env.render()

        out_t = O_net_out.eval(session=sess, feed_dict={O_net_in: [s_t]})  # Get action from online_nn
        a_t = np.zeros([ACTIONS])

        action_index = np.argmax(out_t)
        a_t[action_index] = 1


        obs, reward, done, info = env.step(np.argmax(a_t))

        obs = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
        obs = np.reshape(obs, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis=2)
        s_t1 = np.append(aux_s, obs, axis=2)
        print(np.allclose(s_t, s_t1))
        s_t = s_t1

GAME = "SpaceInvaders-v0"
ENV = gym.make(GAME)
ACTIONS = ENV.action_space.n

O_net_in, O_net_out, Wo_conv1, bo_conv1, Wo_conv2, bo_conv2, Wo_conv3, bo_conv3, Wo_fc, bo_fc, Wo_out, bo_out = networkmaker.Networkmaker(
    ACTIONS).createNetwork()

sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
checkpoint = tf.train.get_checkpoint_state("asyn_network_saves")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Loaded Network: {}".format(checkpoint.model_checkpoint_path))

if __name__ == "__main__":
    play_game()

