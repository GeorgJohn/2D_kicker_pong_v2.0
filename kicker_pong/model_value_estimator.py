import tensorflow as tf

slim = tf.contrib.slim


class ValueEstimator(object):

    def __init__(self, session, state_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4,
                 hidden_size_5, learning_rate=1e-3):
        self.session = session
        self.state_size = state_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.learning_rate = learning_rate

        self.build_model()
        self.build_training()

    def build_model(self):
        with tf.variable_scope('value-model'):
            self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.h0 = slim.fully_connected(self.state, self.hidden_size_1, activation_fn=tf.nn.relu)
            self.h1 = slim.fully_connected(self.h0, self.hidden_size_2, activation_fn=tf.nn.relu)
            self.h2 = slim.fully_connected(self.h1, self.hidden_size_3, activation_fn=tf.nn.relu)
            self.h3 = slim.fully_connected(self.h2, self.hidden_size_4, activation_fn=tf.nn.relu)
            self.h4 = slim.fully_connected(self.h3, self.hidden_size_5, activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(self.h4, 1, activation_fn=None)

    def build_training(self):
        self.reward_input = tf.placeholder(tf.float32, shape=[None])

        self.loss = tf.squared_difference(self.output, self.reward_input)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def predict_value(self, state):
        return self.session.run(self.output, feed_dict={self.state: [state]})[0]
