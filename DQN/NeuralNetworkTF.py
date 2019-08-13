
import tensorflow as tf  # tensorflow (version 1.13.1)

BATCH_SIZE = 10
LEARNING_RATE = 0.001
nn_epsilon = 0.0001


# ========================================= copy Q func to Q target ==========================================

def reset_q_target(sess, q_func, q_target):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(q_func.name)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(q_target.name)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


# ============================================ screen pre process ============================================
"""
1. transform to grey or black-white
2. cut edge with score (optional)
3. resize to 84x84

create screen queue (size=4):
- 4 screen image after pre processing
- if it first screen image - enter to queue 4 identical image
- else dequeue oldest image and enter current
"""


class ProcessImage:
    def __init__(self):
        self.input_img_size = [210, 160, 3]
        self.output_img_size = [84, 84]

        with tf.variable_scope("process_image"):
            self.input_image = tf.placeholder(shape=self.input_img_size, dtype=tf.uint8)
            self.gray_scale = tf.image.rgb_to_grayscale(self.input_image)
            self.crop_img = tf.image.crop_to_bounding_box(self.gray_scale, 34, 0, 160, 160)
            self.resize_img = tf.image.resize_images(
                self.crop_img, self.output_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Removes dimensions of size 1 from the shape of a tensor.
            self.output_image = tf.squeeze(self.resize_img)

    def process(self, sess, state):  # session
        return sess.run(self.output_image, {self.input_image: state})


# ================================================== calc q ==================================================

class NeuralNetwork:
    def __init__(self, name):
        self.input_size = [None, 84, 84, 4]
        self.action_size = 6
        self.name = name

        with tf.variable_scope(name):
            # ======================= Input ============================
            # [None, x1, x2, ...,xn] when x1,...,xn is the size of input features
            # inputs = 4 image of 84*84 = 4 last states
            # ==========================================================
            self.inputs = tf.placeholder(tf.float32, [None, 84, 84, 4], name="inputs")

            # ====================== Action ============================
            # 6 possible action (2*right, 2*left, 2*noop)
            # ==========================================================
            self.actions = tf.placeholder(tf.int32, [None], name="actions")

            # ====================== Q function ========================
            # Q function: r(s,a) + gamma * max Q(s', a')
            # ==========================================================
            self.q_target = tf.placeholder(tf.float32, [None], name="q_target")

            # ================= 1st convolution network=================
            # apply 32 8x8 filters to the input layer
            # ==========================================================
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs, filters=32, kernel_size=[8, 8], strides=[4, 4], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="convolution1")

            # batch_normalization
            self.conv1_batch_norm = tf.layers.batch_normalization(
                self.conv1, training=True, epsilon=nn_epsilon, name='convolution_batch_norm1')

            # ELU
            self.conv1_output = tf.nn.elu(self.conv1_batch_norm, name="convolution_output1")

            # ================= 2nd convolution network=================
            # apply 64 4x4 filters
            # ==========================================================
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_output, filters=64, kernel_size=[4, 4], strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="convolution2")

            # batch_normalization
            self.conv2_batch_norm = tf.layers.batch_normalization(
                self.conv2, training=True, epsilon=nn_epsilon, name='convolution_batch_norm2')

            # ELU
            self.conv2_output = tf.nn.elu(self.conv2_batch_norm, name="convolution_output2")

            # ================= 3rd convolution network=================
            # apply 128 4x4 filters
            # ==========================================================
            # conv2d, padding of one pixel, create 128 images
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_output, filters=128, kernel_size=[4, 4], strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="convolution3")

            # batch_normalization
            self.conv3_batch_norm = tf.layers.batch_normalization(
                self.conv3, training=True, epsilon=nn_epsilon, name='convolution_batch_norm3')

            # ELU
            self.conv3_out = tf.nn.elu(self.conv3_batch_norm, name="convolution_output3")

            # ======================== flatten =========================
            # ==========================================================
            self.flatten = tf.layers.flatten(self.conv3_out)

            # ========================= dense ==========================
            # use the dense() method in layers to connect our dense layer
            # 512 neurons in the dense layer
            # elu activation
            # ==========================================================
            self.dense1 = tf.layers.dense(
                inputs=self.flatten, units=512, activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense1")

            # ========================= output =========================
            # return the raw values for our predictions.
            # We create a dense layer with 6 neurons (one for each target class 0–5)
            # ==========================================================
            self.output = tf.layers.dense(
                inputs=self.dense1, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=6, activation=None)

            # ================== predicted Q value =====================
            # connect output layer and action
            # tf.range = [0,1,...,31]
            # ==========================================================
            self.gather_indices = tf.range(BATCH_SIZE) * tf.shape(self.output)[1] + self.actions
            self.q_func = tf.gather(tf.reshape(self.output, [-1]), self.gather_indices)

            # ========================= loss ===========================
            # The difference between predicted Q values and the Q target
            # loss = Sum(Q_target - Q)^2
            # The loss function will measure how different the output
            # of the network is compared to the target data.
            # ==========================================================
            self.square = tf.square(self.q_target - self.q_func)
            self.loss = tf.reduce_mean(self.square)

            # =================== RMSPropOptimizer =====================
            # two advanced optimization techniques known as RMSprop and Adam.
            # RMSprop and Adam include the concept of momentum (a velocity component).
            # This allows faster convergence at the cost of more computation.

            # model to optimize this loss value during training.
            # We'll use a learning rate of 0.001
            # ==========================================================
            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.output, {self.inputs: state})

    def update(self, sess, state, action, q_target):
        optimizer, loss = sess.run(
            [self.optimizer, self.loss],
            {self.inputs: state, self.q_target: q_target, self.actions: action})
        return loss



