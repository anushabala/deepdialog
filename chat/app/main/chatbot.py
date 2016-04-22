__author__ = 'anushabala'
import random

class ChatBot(object):
    def __init__(self, scenario, agent_num):
        # todo add taggers etc
        self.scenario = scenario
        self.agent_num = agent_num

    def receive(self, message):
        pass # tag sentence and perform some update of probabilities

    def send(self):
        return random.choice(["I'm just a bot that can't really do much right now :)",
                              "Nice to meet you!"])

    def start_chat(self):
        return "hey" # todo choose from template