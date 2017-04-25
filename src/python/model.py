from ModelFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 1000

class Model(ModelInstance):
    def __init__(self):
        self.sess = None
        self._buildGraph();

    def _buildGraph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            self.bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

            self.input = tf.placeholder(tf.float32, shape=(1, batch_size))
            self.output = tf.add(tf.matmul(self.av, self.input), self.bv)

            self.init_op = tf.global_variables_initializer()

    def Inference(self, input):
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init_op)

        with self.sess.as_default():
            return self.sess.run([self.output], feed_dict={self.input: input})

    def SetModelParams(self, params):
        print type(params)
        print params[0]
        # for p in params:
        #     print p
