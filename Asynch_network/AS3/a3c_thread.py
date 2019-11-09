import const
import network
import tensorflow as tf
import numpy as np
import cv2
import gym
from collections import deque

class A3C_Thread:
    def __init__(self, thread_index,  max_glob_t, action_size):
        self.thread_index = thread_index
        self.max_t = max_glob_t
        self.action_size = action_size

        #setup local network
        self.local_network = network.Network(action_size, thread_index)

        self.local_t = 0

        self.episode_reward = 0
        self.s_t = None


    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def get_init_obs(self, obs):
        x_t = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def step(self, sess, global_t, global_network, lock, saver, env):
        start_t = 0
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        #copy weights from shared
        self.local_network.sync_network(global_network, sess, lock)

        for i in range(const.LOCAL_T_MAX):
            pi, value = self.local_network.run_policy_and_value(sess, self.s_t)
            action = self.choose_action(pi)

            states.append(self.s_t)
            actions.append(action)
            values.append(value)

            lock.acquire()
            obs, reward, done, info = env.step(action)
            lock.release()

            self.episode_reward += reward

            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1
            start_t += 1

            x_t1 = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            aux_s = np.delete(self.s_t, 0, axis=2)
            self.s_t = np.append(aux_s, x_t1, axis=2)

            if done:
                env.reset()
                obs = env.reset()
                x_t1 = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                aux_s = np.delete(self.s_t, 0, axis=2)
                self.s_t = np.append(aux_s, x_t1, axis=2)

                print("THREAD:{} | REWARD:{} | TIME:{} ".format(self.thread_index, self.episode_reward, self.local_t ))
                self.episode_reward = 0
                terminal_end = True
                break

        R = 0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + const.GAMMA * R
            td = R - Vi
            a = np.zeros([self.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if self.local_t % 5000 == 0:
            saver.save(sess, 'a3c_saves/a3c-dqn', global_step=global_t)

        return self.local_t - start_t, (batch_si, batch_a, batch_td, batch_R)