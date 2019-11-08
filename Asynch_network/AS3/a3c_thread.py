import const
import network
import tensorflow as tf
import numpy as np
import cv2
import gym
from collections import deque

class A3C_Thread:
    def __init__(self, thread_index, init_lr, lr, grad_applier, max_glob_t, action_size):
        self.thread_index = thread_index
        self.lr = lr
        self.max_t = max_glob_t
        self.action_size = action_size

        #setup local network
        self.local_network = network.Network(action_size, thread_index)
        self.local_network.prepare_loss(const.ENTROPY_BETA)

        self.apply_gradients = grad_applier

        self.local_t = 0

        self.initial_learning_rate = init_lr

        self.episode_reward = 0
        self.s_t = None

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
                    self.max_t - global_time_step) / self.max_t
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def get_init_obs(self, obs):
        obs = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
        self.s_t = deque([obs, obs, obs, obs])

    def step(self, sess, global_t, global_network, lock, saver, env):
        start_t = 0
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        #copy weights from shared
        self.local_network.sync_from(global_network)

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
            self.s_t.pop()
            self.s_t.appendleft(x_t1)

            if done:
                env.reset()
                obs = env.reset()
                x_t1 = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                self.s_t.pop()
                self.s_t.appendleft(x_t1)
                
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

        cur_learning_rate = self._anneal_learning_rate(global_t)

        if self.local_t % 5000 == 0:
            saver.save(sess, 'a3c_saves/a3c-dqn', global_step=global_t)

        return self.local_t - start_t, (batch_si, batch_a, batch_td, batch_R), cur_learning_rate