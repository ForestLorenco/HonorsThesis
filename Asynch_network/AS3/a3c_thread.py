import const
import network
import tensorflow as tf
import numpy as np
import cv2
import gym

class A3C_Thread:
    def __init__(self, thread_index, global_network, init_lr, lr, grad_applier, max_glob_t, action_size):
        self.thread_index = thread_index
        self.lr = lr
        self.max_t = max_glob_t
        self.action_size = action_size

        #setup local network
        self.local_network = network.Network(action_size, thread_index)
        self.local_network.prepare_loss(const.ENTROPY_BETA)

        var_refs = [v._ref() for v in self.local_network.get_vars()]
        self.gradients = tf.gradients(
            self.local_network.total_loss, var_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.local_t = 0

        self.initial_learning_rate = init_lr

        self.episode_reward = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
                    self.max_t - global_time_step) / self.max_t
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def step(self, sess, global_t, global_network, lock, saver):
        start_t = 0
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        lock.acquire()
        env = gym.make(const.GAME)
        obs = env.reset()
        lock.release()

        obs = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (80, 80))
        s_t = np.stack((obs,obs,obs,obs), axis=2)
        aux_s = s_t

        #copy weights from shared
        self.local_network.sync_from(global_network)

        for i in range(const.LOCAL_T_MAX):
            pi, value = self.local_network.run_policy_and_value(sess, s_t)
            action = self.choose_action(pi)

            states.append(s_t)
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
            aux_s = np.delete(s_t, 0, axis=2)
            s_t1 = np.append(aux_s, x_t1, axis=2)

            s_t = s_t1

            if done:
                env.reset()
                print("THREAD:{} | STATE:{} | REWARD:{} | TIME:{} | EPSILON: {}".format(id, state, score, t, epsilon))
                self.episode_reward = 0
                terminal_end = True
                break

        R = 0
        if not terminal_end:
            R = self.local_network.run_value(sess, s_t)

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

        sess.run(self.apply_gradients,
                 feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R,
                     self.lr: cur_learning_rate})

        if self.local_t % 5000 == 0:
            saver.save(sess, 'a3c_saves/a3c-dqn', global_step=t)

        return self.local_t - start_t