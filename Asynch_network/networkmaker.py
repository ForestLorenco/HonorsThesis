import tensorflow as tf


class Networkmaker:
    def __init__(self, action_size):
        self.action_size = action_size

        # things for the conv network
        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8, 8, 4), (4, 4, 32), (3, 3, 64)]
        self.conv_strides = [4, 2, 1]
        self.input_height = 80
        self.input_width = 80
        self.input_channels = 4

    def weight_var(self, shape):
        """
        Creates a tensor that is shape with random numbers from a normal distritution with stdev 0.01
        :param shape:
        :return a tf variable of shape shape:
        """
        init = tf.random.truncated_normal(shape, stddev=0.01)
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
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def createNetwork(self):
        """
        Creates and returns DQN network for agent
        :return output and input layer, as well as all weight and bias tensors for nn:
        """
        # input layer
        inp_layer = tf.compat.v1.placeholder("float", [None, self.input_height, self.input_width, self.input_channels])

        # first convolutional layer tensors
        W_conv1 = self.weight_var([8, 8, 4, 32])
        b_conv1 = self.bias_var([32])

        W_conv2 = self.weight_var([4, 4, 32, 64])
        b_conv2 = self.bias_var([64])

        W_conv3 = self.weight_var([3, 3, 64, 64])
        b_conv3 = self.bias_var([64])

        # Fully connected layer tensors
        W_fc = self.weight_var([256, 256])
        b_fc = self.bias_var([256])

        # ouptput layer tensors
        W_out = self.weight_var([256, self.action_size])
        b_out = self.bias_var([self.action_size])

        # Hidden layer declerations
        h_conv1 = tf.nn.relu(self.conv2d(inp_layer, W_conv1, self.conv_strides[0]) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, self.conv_strides[1]) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, self.conv_strides[2]) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        # Flatten layer
        h_pool3 = tf.reshape(h_pool3, [-1, 256])

        h_fc = tf.nn.relu(tf.matmul(h_pool3, W_fc) + b_fc)

        output = tf.matmul(h_fc, W_out) + b_out

        return inp_layer, output, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc, b_fc, W_out, b_out
