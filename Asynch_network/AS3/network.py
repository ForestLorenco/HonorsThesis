import tensorflow as tf
import numpy as np

class Network:
    def __init__(self,action_size, thread_index):
        self.action_size = action_size
        self.thread_index = thread_index

    def fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

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
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    def make_network(self):
        self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 16])  # stride=4
        self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 16, 32])  # stride=2

        self.W_fc1, self.b_fc1 = self.fc_variable([2592, 256])

        # weight for policy output layer
        self.W_fc2, self.b_fc2 = self.fc_variable([256, self.action_size])

        # weight for value output layer
        self.W_fc3, self.b_fc3 = self.fc_variable([256, 1])

        # state (input)
        self.s = tf.placeholder("float", [None, 80, 80, 4])

        h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

        # policy (output)
        self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        # value (output)
        v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
        self.v = tf.reshape(v_, [-1])

    def prepare_loss(self, e_beta):
        #Action input for policy
        self.a = tf.compat.v1.placeholder("float", [None, self.action_size])

        self.temp_diff = tf.compat.v1.placeholder("float", [None, self.action_size])

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

        # policy entropy
        entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
        policy_loss = - tf.reduce_sum(
            tf.reduce_sum(tf.multiply(log_pi, self.a), reduction_indices=1) * self.temp_diff + entropy * e_beta)

        #input for the value function
        self.r = tf.compat.v1.placeholder("float", [None])

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

        # gradienet of policy and value are summed up
        self.total_loss = policy_loss + value_loss

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]

    def sync_network(self, src_network, sess):
        """
        Copies the target network to the thread specific network
        :param src_network:
        :param sess:
        :return:
        """
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()
        copy_Otarget = []

        for src_var, dst_var in zip(src_vars, dst_vars):
            copy_Otarget.append(src_var.assign(dst_var))

        sess.run(copy_Otarget)