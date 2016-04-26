__author__ = 'anushabala'
import random
import datetime
import numpy as np
from collections import defaultdict
import operator
from tagger import Entity, Template, TemplateType
import time


def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)


class ChatState(object):
    CHAT=1
    ASK=2
    TELL=3
    SUGGEST=4
    SELECT_GUESS=5
    SELECT_FINAL=6
    FINISHED=7


class ChatBot(object):
    # todo change this later so tagger returns confidences
    FULL_NAME_BOOST = 0.5
    PROB_BOOST_DIRECT_MENTION = 0.02
    PROB_BOOST_SYNONYM = 0.01
    CHAR_RATE = 20
    SELECTION_DELAY = 2
    EPSILON = 1

    def __init__(self, scenario, agent_num, tagger):
        self.scenario = scenario
        self.agent_num = agent_num
        self.friends = scenario["agents"][agent_num]["friends"]
        self.my_info = scenario["agents"][agent_num]["info"]
        self.my_turn = True
        self.tagger = tagger
        self.state = None
        self.last_message_timestamp = datetime.datetime.now()
        self.probabilities = {}

        self.full_names_cased = {}
        self.school_to_friends = defaultdict(list)
        self.major_to_friends = defaultdict(list)
        self.company_to_friends = defaultdict(list)
        self.first_names_to_friends = defaultdict(list)
        self.ranked_friends = [friend["name"] for friend in self.friends]
        self.init_probabilities()
        self.create_mappings()
        self.friend_ctr= 0
        self.selection = None
        self.next_text = None

    def init_probabilities(self):
        for friend in self.friends:
            self.probabilities[friend["name"].lower()] = 1.0/len(self.friends)


    def rerank_friends(self):
        self.ranked_friends = [x[0] for x in sorted(self.probabilities.items(),
                                                    key=operator.itemgetter(1),
                                                    reverse=True)]
        print sorted(self.probabilities.items(), key=operator.itemgetter(1), reverse=True)
        self.friend_ctr = 0

    def create_mappings(self):
        for friend in self.friends:
            name = friend["name"]
            first_name = name.split()[0]
            self.full_names_cased[name.lower()] = name
            self.school_to_friends[friend["school"]["name"].lower()].append(name.lower())
            self.major_to_friends[friend["school"]["major"].lower()].append(name.lower())
            self.company_to_friends[friend["company"]["name"].lower()].append(name.lower())
            self.first_names_to_friends[first_name.lower()].append(name.lower())
        print self.school_to_friends
        print self.major_to_friends
        print self.company_to_friends
        print self.first_names_to_friends

    def set_final_probability(self, name):
        for friend in self.probabilities.keys():
            if friend == name:
                self.probabilities[friend] = 1.0
            else:
                self.probabilities[friend] = 0.0

    def update_probabilities(self, found_entities):
        print found_entities
        print len(found_entities.keys())
        if found_entities is None or len(found_entities.keys()) == 0:
            return
        for entity_type in found_entities.keys():
            entities = found_entities[entity_type]
            print entity_type, entities
            if entity_type == Entity.to_str(Entity.FULL_NAME):
                for entity in entities:
                    if entity in self.probabilities.keys():
                        print "Full name boost %2.2f for %s" % (self.FULL_NAME_BOOST, entity)
                        self.probabilities[entity] += self.FULL_NAME_BOOST
            else:
                mapping = self.first_names_to_friends
                if entity_type == Entity.to_str(Entity.COMPANY_NAME):
                    mapping = self.company_to_friends
                elif entity_type == Entity.to_str(Entity.MAJOR):
                    mapping = self.major_to_friends
                elif entity_type == Entity.to_str(Entity.SCHOOL_NAME):
                    mapping = self.school_to_friends

                for entity in entities:
                    for friend in mapping[entity]:
                        print "<%s> Direct mention boost for %s" % (entity, friend)
                        self.probabilities[friend] += self.PROB_BOOST_DIRECT_MENTION

        items = self.probabilities.items()
        raw_probabilities = softmax([item[1] for item in items])
        self.probabilities = {items[i][0]:raw_probabilities[i] for i in range(0, len(items))}
        self.rerank_friends()

    def receive(self, message):
        self.last_message_timestamp = datetime.datetime.now()
        self.my_turn = True
        found_entities = self.tagger.tag_sentence(message)
        # todo set state to select and update probabilities if partner selects the mutual friend
        # todo low priority maybe also say "no i don't know them" + next message if selection isn't mutual friend
        self.update_probabilities(found_entities)

    def partner_selection(self, selection):
        print "<partner selection>", selection
        if selection.lower() in self.probabilities.keys():
            self.state = ChatState.SELECT_FINAL
            self.my_turn = True
            self.set_final_probability(selection.lower())
            self.rerank_friends()
        else:
            # todo send message "no i don't know"
            pass

    def send(self):
        if self.next_text is not None:
                delay = float(len(self.next_text))/self.CHAR_RATE
                if self.last_message_timestamp + datetime.timedelta(milliseconds=delay*1000) > datetime.datetime.now():
                    return None, None
                else:
                    ret_text = self.next_text
                    self.next_text = None
                    self.last_message_timestamp = datetime.datetime.now()
                    print ret_text
                    return None, ret_text.lower()

        if not self.my_turn:
            return None, None
        else:
            selection = None
            if self.state is None:
                self.state = ChatState.CHAT
            elif self.state == ChatState.FINISHED:
                self.my_turn = False
                return selection, None
            elif self.state == ChatState.CHAT:
                self.state = ChatState.TELL
            elif self.state == ChatState.TELL:
                self.my_turn = False
                self.state = ChatState.ASK
            elif self.state == ChatState.ASK:
                self.my_turn = False
                self.state = ChatState.SUGGEST
            elif self.state == ChatState.SUGGEST:
                self.my_turn = False
                # some logic here to make a selection - threshold probability?
            elif self.state == ChatState.SELECT_FINAL:
                self.state = ChatState.FINISHED
                self.my_turn = False
                self.selection = selection = self.ranked_friends[0]
                return self.full_names_cased[self.selection], None

            self.next_text = self.generate_text()
            delay = float(len(self.next_text)/self.CHAR_RATE)
            if self.last_message_timestamp + datetime.timedelta(days=0, seconds=0, milliseconds=delay*1000) > datetime.datetime.now():
                return None, None
            else:
                ret_text = self.next_text
                self.next_text = None
                self.last_message_timestamp = datetime.datetime.now()
                print ret_text
                return selection, ret_text.lower()

    def generate_text(self):
        if self.state == ChatState.CHAT:
            text = self.tagger.get_template(TemplateType.CHAT).text
        elif self.state == ChatState.ASK:
            text = self.tagger.get_template(TemplateType.ASK).text
        elif self.state == ChatState.TELL:
            template = self.tagger.get_template(TemplateType.TELL)
            text = self.fill_in_template(template)
        elif self.state == ChatState.SUGGEST:
            template = self.tagger.get_template(TemplateType.SUGGEST)
            text = self.fill_in_template(template)
        else:
            text = None

        return text

    def fill_in_template(self, template):
        entities = template.get_entities()
        named_entities = []
        ctr = self.friend_ctr
        for entity in entities:
            if entity == '<schoolName>':
                named_entities.append(self.my_info["school"]["name"])
            elif entity == '<major>':
                named_entities.append(self.my_info["school"]["major"])
            elif entity == '<companyName>':
                named_entities.append(self.my_info["company"]["name"])
            else:
                if ctr < len(self.ranked_friends):
                    friend = self.ranked_friends[ctr]
                else:
                    ctr = 0
                    self.friend_ctr = 0
                    friend = self.ranked_friends[ctr]
                ctr += 1
                if entity == '<firstName>':
                    named_entities.append(friend.split()[0])
                elif entity == '<fullName>':
                    named_entities.append(friend)
        self.friend_ctr = ctr + 1
        return template.fill(entities, named_entities)

    def start_chat(self):
        return "hey" # todo choose from template