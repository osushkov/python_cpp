from StrategyFramework import *
import math

class Strategy(StrategyInstance):
    def __init__(self):
        self.sum = 0.0

    def eval(self, vals):
        print "strategy.eval"
        self.sum += math.sqrt(2.0)
        print self.sum

        for v in vals:
            print v
