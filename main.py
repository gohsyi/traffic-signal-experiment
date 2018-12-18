import tensorflow as tf
import numpy as np
import argparse
from OUTPUT import Output
from collect import *

reward_mat_row = np.array([
    -1, -1, 1, -10
])

reward_mat_col = np.array([
    -1, 1, -1, -10
])


def parse_arg():

    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('-history_length', type=int, default=10)
    parser.add_argument('-sig_type', type=str, default='normal')
    parser.add_argument('-sig_size', type=int, default=1)
    parser.add_argument('-hid_size', type=str, default='')
    parser.add_argument('-ac_fn', type=str, default='tanh')  # relu, tanh
    parser.add_argument('-batch_size', type=int, default=3*int(1e4))
    parser.add_argument('-use_bias', type=bool, default=True)

    # training settings
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-max_episode', type=int, default=int(1e5))

    # IO settings
    parser.add_argument('-eval_interval', type=int, default=1000)
    parser.add_argument('-eval_number', type=int, default=1000)
    parser.add_argument('-output_path', type=str, default='log')
    parser.add_argument('-restore_path', type=str, default='log')
    parser.add_argument('-note', type=str, default='')

    args = parser.parse_args()

    return args


class Model:

    def __init__(self, args):
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        self.sig_type = args.sig_type
        self.sig_size = args.sig_size
        self.hid_size = list(map(int, args.hid_size.split(','))) if len(args.hid_size) else []
        self.max_episode = args.max_episode
        self.history_length = args.history_length

        if args.ac_fn == 'tanh':
            self.fn = tf.nn.tanh
        elif args.ac_fn == 'relu':
            self.fn = tf.nn.relu
        elif args.ac_fn == 'sigmoid':
            self.fn = tf.nn.sigmoid
        elif args.ac_fn == 'elu':
            self.fn = tf.nn.elu
        else:
            raise ValueError

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_number = args.eval_number
        self.note = args.note
        self.restore_path = args.restore_path
        self.output = Output(args)
        self.wr = self.output.write
        self.use_bias = args.use_bias

        # RNN
        self.lstm_hidden_size = 1
        # self acts, self rewards, opponent acts, opponent rewards
        self._history_row = np.zeros(dtype=np.float32, shape=(self.batch_size, self.history_length))
        self._history_col = np.zeros(dtype=np.float32, shape=(self.batch_size, self.history_length))
        self.history_row = tf.get_variable(name="history_row", dtype=tf.float32,
                                           shape=[self.batch_size, self.history_length])
        self.history_col = tf.get_variable(name="history_col", dtype=tf.float32,
                                           shape=[self.batch_size, self.history_length])
        if self.sig_type == 'rnn': self.eval_interval = 1; self.eval_number = self.batch_size

        self.output.debug_write(str(args))
        self.sess = tf.Session()
        self.setup_env()
        self.learn()


    def generate_data(self, new_act_row, new_act_col):
        self._history_row[:, :-1] = self._history_row[:, 1:]
        self._history_row[:, -1] = new_act_col

        self._history_col[:, :-1] = self._history_col[:, 1:]
        self._history_col[:, -1] = new_act_row

        self.wr(str(self._history_row[0, :]))


    def setup_env(self):
        self.lr_ = tf.placeholder(name='lr', dtype=tf.float32, shape=[])

        with tf.variable_scope('row_player'):
            if self.sig_type == 'rnn':
                self.pi_row_ = self.build_rnn(self.history_row, 1)
            else:
                self.sig_row_ = tf.placeholder(name='sig_row', dtype=tf.float32, shape=[self.batch_size, self.sig_size])
                self.pi_row_ = self.agt(self.sig_row_, 1)

        with tf.variable_scope('col_player'):
            if self.sig_type == 'rnn':
                self.pi_col_ = self.build_rnn(self.history_col, 2)
            else:
                self.sig_col_ = tf.placeholder(name='sig_col', dtype=tf.float32, shape=[self.batch_size, self.sig_size])
                self.pi_col_ = self.agt(self.sig_col_, 2)

        self.pi_mat_ = tf.reshape(tf.expand_dims(self.pi_row_, 2) * tf.expand_dims(self.pi_col_, 1), [self.batch_size, -1])
        self.loss_row_ = -tf.reduce_mean(self.pi_mat_*reward_mat_row)
        self.loss_col_ = -tf.reduce_mean(self.pi_mat_*reward_mat_col)

        self.train_op_row_ = tf.train.GradientDescentOptimizer(self.lr_).minimize(self.loss_row_,
                var_list=[v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='row_player')])

        self.train_op_col_ = tf.train.GradientDescentOptimizer(self.lr_).minimize(self.loss_col_,
                var_list=[v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='col_player')])

        tf.summary.FileWriter("logs", tf.get_default_graph()).close()


    def learn(self):

        self.sess.run(tf.global_variables_initializer())

        for ep in range(self.max_episode):
            sig = np.zeros(shape=[self.batch_size, self.sig_size], dtype=np.float)
            if self.sig_type == 'normal':
                sig = np.random.normal(size=[self.batch_size, self.sig_size])
            elif self.sig_type == 'uniform':
                sig = np.random.uniform(low=-1, high=1, size=[self.batch_size, self.sig_size])
            elif self.sig_type == 'onehot':
                sig = np.eye(self.sig_size)[np.random.choice(self.sig_size, size=self.batch_size)]
            elif self.sig_type != 'rnn':
                raise('signal type not implemented')

            sig_row = sig
            sig_col = sig

            if self.sig_type == 'rnn':
                feed_dict = {
                    self.history_row : self._history_row,
                    self.history_col : self._history_col,
                    self.lr_ : (1 - ep/self.max_episode) * self.lr
                }
            else:
                feed_dict = {
                    self.sig_row_: sig_row,
                    self.sig_col_: sig_col,
                    self.lr_ : (1 - ep/self.max_episode) * self.lr
                }

            if ep % self.eval_interval == 0:
                pi_row, pi_col = self.sess.run([self.pi_row_, self.pi_col_], feed_dict=feed_dict)
                pi_row_, pi_col_, sig_row, sig_col = \
                    np.argmax(pi_row, axis=1)[:self.eval_number], np.argmax(pi_col, axis=1)[:self.eval_number], \
                    sig_row[:self.eval_number], sig_col[:self.eval_number]

                if self.sig_type == 'rnn':
                    self.generate_data(pi_row_, pi_col_)
                else:
                    for i in range(self.eval_number):
                        self.wr('[Eval:%i]|%s|%s|%s|%s' % (ep, str(list(sig_row[i])), str(list(sig_col[i])), str(pi_row_[i]), str(pi_col_[i])))
                        print(pi_row[i], pi_col[i])

            _, _, log_loss_row, log_loss_col = self.sess.run(
                [self.train_op_row_, self.train_op_col_, self.loss_row_, self.loss_col_],
                feed_dict=feed_dict
            )
            wr_interval = self.eval_interval//20 if self.eval_interval >= 20 else 1
            if ep % wr_interval == 0:
                self.wr('[Train:%i]loss_row: %f' % (ep, log_loss_row))
                self.wr('[Train:%i]loss_col: %f' % (ep, log_loss_col))


    def agt(self, sig, no):
        last_out = sig

        for i in range(len(self.hid_size)):
            last_out = self.fn(tf.layers.dense(last_out, self.hid_size[i], name="fc%i_%i" % (i + 1, no), use_bias=self.use_bias,
                                          kernel_initializer=tf.random_normal_initializer(mean=0)))
        pi_ = tf.layers.dense(
            inputs=last_out,
            units=2,
            name='ac_%i' % no,
            use_bias=self.use_bias,
            kernel_initializer=tf.random_normal_initializer(mean=0)
        )

        return tf.nn.softmax(pi_, axis=1)


    def build_rnn(self, history, no):
        lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, name='lstm_{}'.format(no))
        pi_ = None
        state = lstm.zero_state(self.batch_size, tf.float32)

        for i in range(self.history_length):
            if i > 0: tf.get_variable_scope().reuse_variables()
            history_slice = history[:, i][:, np.newaxis]
            lstm_output, state = lstm(history_slice, state)
            pi_ = self.fn(tf.layers.dense(lstm_output, 2, name='ac_%i' % no, use_bias=self.use_bias,
                                          kernel_initializer=tf.random_normal_initializer(mean=0)))

        return tf.nn.softmax(pi_, axis=1)


if __name__ == '__main__':

    args = parse_arg()
    # print(args)

    model = Model(args)
    # collect(args)
