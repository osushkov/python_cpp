from ModelFramework import *
import numpy as np
import tensorflow as tf


class Model(ModelInstance):
# class Model:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sess = None
        self.params = [np.array([[5.0]]), np.array([[7.0]])]
        self._buildGraph();


    def Inference(self, input):
        self._initSession()
        with self.sess.as_default():
            return self.sess.run([self.output], feed_dict={self.input: input.reshape(1, self.batch_size),
                                                           self.av: self.params[0],
                                                           self.bv: self.params[1]})[0]


    def SetModelParams(self, params):
        self.params = params


    def _buildGraph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.av =  tf.placeholder(tf.float32, shape=(1, 1))
            self.bv =  tf.placeholder(tf.float32, shape=(1, 1))

            self.input = tf.placeholder(tf.float32, shape=(1, self.batch_size))
            self.output = tf.add(tf.matmul(self.av, self.input), self.bv)

            self.init_op = tf.global_variables_initializer()


    def _initSession(self):
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init_op)
