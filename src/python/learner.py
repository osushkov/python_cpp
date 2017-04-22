
# from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createSampleData(samples):
    a = 10.0;
    b = -5.0;
    noise_sd = 1.0

    xs = np.random.rand(1, samples) * 1.0
    ys = xs * a + b + np.random.normal(0.0, noise_sd, samples)

    return xs, ys


def makeBatch(batch_size, data_x, data_y):
    indices = np.random.permutation(data_x.shape[1])[:batch_size]
    return data_x[:,indices], data_y[:,indices]


# plt.scatter(xs, ys)
# plt.show()

batch_size = 1000

graph = tf.Graph()
with graph.as_default():
    av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
    bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

    xv = tf.placeholder(tf.float32, shape=(1, batch_size))
    yv = tf.placeholder(tf.float32, shape=(1, batch_size))
    ypred = tf.add(tf.matmul(av, xv), bv)

    loss = tf.reduce_mean(tf.squared_difference(ypred, yv))
    opt = tf.train.AdamOptimizer().minimize(loss)

    init_op = tf.global_variables_initializer()

sess = tf.Session(graph=graph)
with sess.as_default():
    sess.run(init_op)

    data_x, data_y = createSampleData(10000)
    for i in range(10000):
        batch_x, batch_y = makeBatch(batch_size, data_x, data_y)
        _, l, a, b = sess.run([opt, loss, av, bv], feed_dict={xv: batch_x, yv: batch_y})
        print("iter: " + str(i) + " loss: " + str(l))
    print(a, b)

batch_size = 1000

class Learner(LearnerInstance):
    def __init__(self):
        self.graph = None
        self.sess = None

        self.data_x, self.data_y = createSampleData(10000)

    def BuildGraph(self):
        print "BuildGraph called"

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            self.bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

            # self.xv = tf.placeholder(tf.float32, shape=(1, batch_size))
            # self.yv = tf.placeholder(tf.float32, shape=(1, batch_size))
            # self.ypred = tf.add(tf.matmul(self.av, self.xv), self.bv)
            #
            # self.loss = tf.reduce_mean(tf.squared_difference(self.ypred, self.yv))
            # self.opt = tf.train.AdamOptimizer().minimize(self.loss)


    def LearnIterations(self, iters):
        sess = tf.Session(graph=self.graph)

        with sess.as_default():
            sess.run(tf.global_variables_initializer())
        # if self.sess is None:
        #     self.sess = tf.Session(graph=self.graph)
        #     self.sess.run(tf.global_variables_initializer())

        # batch_x, batch_y = makeBatch(batch_size, self.data_x, self.data_y)
        # _, l, a, b = sess.run([self.opt, self.loss, self.av, self.bv],
        #                       feed_dict={self.xv: batch_x, self.yv: batch_y})
        # print("iter: " + str(i) + " loss: " + str(l))
