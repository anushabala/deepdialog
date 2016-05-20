"""A generic continuous neural sequence-to-sequence model."""
import collections
import numpy
import os
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import time


class TrainingListener(object):
    def __init__(self):
        pass

    def start_training(self,model):
        pass

    def finish_training(self,model):
        pass

    def start_training_epoch(self,i,model):
        pass

    def finish_training_epoch(self,i,model):
        pass


class EvaluateEveryEpoch(TrainingListener):
    def __init__(self, devset, testset):
        pass

    def start_training(self,model):
        pass

    def finish_training(self,model):
        pass

    def start_training_epoch(self,i,model):
        pass

    def finish_training_epoch(self,i,model):
        pass


