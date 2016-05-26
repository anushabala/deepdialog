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
    return exp_x / np.sum(exp_x)


class ChatState(object):
    CHAT = 1
    ASK = 2
    TELL = 3
    SUGGEST = 4
    SELECT_GUESS = 5
    SELECT_FINAL = 6
    ACCEPT = 7
    REJECT = 8
    FINISHED = 9


class ChatBotBase(object):
    probabilities = {}
    name = "DEFAULT_BOT"
    def start(self):
        raise NotImplementedError

    def init_probabilities(self):
        raise NotImplementedError

    def receive(self, message):
        raise NotImplementedError

    def partner_selection(self, selection):
        raise NotImplementedError

    def send(self):
        raise NotImplementedError

    def end_chat(self):
        raise NotImplementedError

    def start_chat(self):
        raise NotImplementedError

    def _get_probability_string(self):
        sorted_probs = sorted(self.probabilities.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
        s = ""
        for (key, value) in sorted_probs:
            s += "(%s, %2.2f),  " % (key, value)

        s = s.strip(',')
        return s

    def _get_entities_string(self, entities):
        s = ""
        for entity_type in entities.keys():
            s += "[%s:" % entity_type
            for ent in entities[entity_type]:
                s += " %s," % ent
            s = s.strip(',')
            s += "],"
        s = s.strip(',')
        return s


class ChatBot(ChatBotBase):
    # todo change this later so tagger returns confidences
    FULL_NAME_BOOST = 1
    PROB_BOOST_DIRECT_MENTION = 1
    PROB_BOOST_SYNONYM = 0.5
    CHAR_RATE = 9.5
    SELECTION_DELAY = 1000
    EPSILON = 1000

    def __init__(self, scenario, agent_num, tagger, name='DEFAULT_BOT'):
        self.scenario = scenario
        self.agent_num = agent_num
        self.friends = scenario["agents"][agent_num]["friends"]
        self.my_info = scenario["agents"][agent_num]["info"]
        self.my_turn = True
        self.tagger = tagger
        self.name = name

        self.state = None
        self.last_message_timestamp = datetime.datetime.now()
        self.probabilities = {}
        self.mentioned_friends = set()

        self.full_names_cased = {}
        self.school_to_friends = defaultdict(list)
        self.major_to_friends = defaultdict(list)
        self.company_to_friends = defaultdict(list)
        self.first_names_to_friends = defaultdict(list)
        self.ranked_friends = [friend["name"] for friend in self.friends]
        self.init_probabilities()
        self.create_mappings()
        self.friend_ctr = 0
        self.selection = None
        self.next_text = None
        self.text_addition = None
        self.template_type = None

    def start(self):
        self.last_message_timestamp = datetime.datetime.now()


    def init_probabilities(self):
        for friend in self.friends:
            self.probabilities[friend["name"].lower()] = 1.0 / len(self.friends)

    def rerank_friends(self):
        self.ranked_friends = [x[0] for x in sorted(self.probabilities.items(),
                                                    key=operator.itemgetter(1),
                                                    reverse=True)]

    def create_mappings(self):
        for friend in self.friends:
            name = friend["name"]
            first_name = name.split()[0]
            self.full_names_cased[name.lower()] = name
            self.school_to_friends[friend["school"]["name"].lower()].append(name.lower())
            self.major_to_friends[friend["school"]["major"].lower()].append(name.lower())
            self.company_to_friends[friend["company"]["name"].lower()].append(name.lower())
            self.first_names_to_friends[first_name.lower()].append(name.lower())

    def set_final_probability(self, name):
        for friend in self.probabilities.keys():
            if friend == name:
                self.probabilities[friend] = 1.0
            else:
                self.probabilities[friend] = 0.0

    def update_probabilities(self, found_entities, guess=False):
        if found_entities is None or len(found_entities.keys()) == 0:
            return
        for entity_type in found_entities.keys():
            entities = found_entities[entity_type]
            if entity_type == Entity.to_str(Entity.FULL_NAME):
                for entity in entities:
                    if entity in self.probabilities.keys():
                        self.probabilities[entity] += self.FULL_NAME_BOOST
                        self.state = ChatState.ACCEPT
                        self.selection = self.full_names_cased[entity]
            else:
                mapping = self.first_names_to_friends
                if entity_type == Entity.to_str(Entity.COMPANY_NAME):
                    mapping = self.company_to_friends
                elif entity_type == Entity.to_str(Entity.MAJOR):
                    mapping = self.major_to_friends
                elif entity_type == Entity.to_str(Entity.SCHOOL_NAME):
                    mapping = self.school_to_friends
                # print "found entities for type: %s" % entity_type, entities
                for entity in entities:
                    # print entity
                    # print mapping[entity]
                    # print "----"
                    for friend in mapping[entity]:
                        if friend in self.probabilities.keys():
                            if not guess:
                                self.probabilities[friend] += self.PROB_BOOST_DIRECT_MENTION
                                if entity_type == Entity.FIRST_NAME:
                                    self.state = ChatState.ACCEPT
                                    self.selection = self.full_names_cased[friend]
                            else:
                                self.probabilities[friend] += self.PROB_BOOST_SYNONYM

        items = self.probabilities.items()
        raw_probabilities = softmax([item[1] for item in items])
        self.probabilities = {items[i][0]: raw_probabilities[i] for i in range(0, len(items))}
        self.rerank_friends()

    def receive(self, message):
        self.last_message_timestamp = datetime.datetime.now()
        self.my_turn = True
        found_entities, possible_entities = self.tagger.tag_sentence(message)
        # print "synonyms from tagger:", possible_entities
        self.update_probabilities(found_entities)
        self.update_probabilities(possible_entities, guess=True)
        ret_data = {"probs": self._get_probability_string(),
                    "confident_tags": self._get_entities_string(found_entities),
                    "possible_tags": self._get_entities_string(possible_entities)}
        return ret_data

    def partner_selection(self, selection):
        self.last_message_timestamp = datetime.datetime.now()
        if selection.lower() in self.probabilities.keys():
            self.state = ChatState.SELECT_FINAL
            self.my_turn = True
            self.set_final_probability(selection.lower())
            self.rerank_friends()
        else:
            self.my_turn = True
            self.state = ChatState.REJECT

    def end_chat(self):
        self.state = ChatState.FINISHED

    def send(self):
        # if accept or reject, try to return immediately
        if self.state == ChatState.ACCEPT or self.state == ChatState.REJECT:
            if self.next_text is None:
                self.next_text = self.generate_text()

        if self.selection is not None and self.state != ChatState.ACCEPT and self.state != ChatState.REJECT:
            delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
            if self.last_message_timestamp + datetime.timedelta(days=0, seconds=0,
                                                                milliseconds=delay) > datetime.datetime.now():
                return None, None
            else:
                selection = self.selection
                self.last_message_timestamp = datetime.datetime.now()
                self.selection = None
                if self.state == ChatState.SELECT_FINAL:
                    self.state = ChatState.FINISHED
                    self.my_turn = False
                elif self.state == ChatState.SELECT_GUESS:
                    self.state = ChatState.SUGGEST
                    self.my_turn = False
                return selection, None

        if self.next_text is not None:
            delay = float(len(self.next_text)) / self.CHAR_RATE * 1000 + self.EPSILON
            if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
                return None, None
            else:
                ret_text = self.next_text
                self.next_text = None
                self.last_message_timestamp = datetime.datetime.now()
                if self.state == ChatState.ACCEPT:
                    if self.selection is None:
                        self.state = ChatState.SELECT_FINAL
                    else:
                        self.state = ChatState.SELECT_GUESS
                elif self.state == ChatState.REJECT:
                    self.state = ChatState.SUGGEST
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
                self.template_type = random.choice(TemplateType.subtypes(TemplateType.TELL)[0:2])
            elif self.state == ChatState.TELL:
                if self.template_type == TemplateType.subtypes(TemplateType.TELL)[1]:
                    self.template_type = TemplateType.subtypes(TemplateType.TELL)[2]
                else:
                    self.template_type = None
                    self.state = ChatState.ASK
                    self.my_turn = False
            elif self.state == ChatState.ASK:
                self.my_turn = False
                self.state = ChatState.SUGGEST
            elif self.state == ChatState.SUGGEST:
                self.my_turn = False
                # some logic here to make a selection - threshold probability?
            elif self.state == ChatState.SELECT_GUESS:
                self.my_turn = False
                self.state = ChatState.SUGGEST
                if self.selection is None:
                    self.selection = selection = self.full_names_cased[self.ranked_friends[0]]
            elif self.state == ChatState.SELECT_FINAL:
                self.state = ChatState.FINISHED
                self.my_turn = False
                self.selection = selection = self.full_names_cased[self.ranked_friends[0]]

            if selection is not None:
                # make selection
                self.next_text = None
                delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
                if self.last_message_timestamp + datetime.timedelta(days=0, seconds=0,
                                                                    milliseconds=delay) > datetime.datetime.now():
                    return None, None
                else:
                    self.last_message_timestamp = datetime.datetime.now()
                    self.selection = None
                    return selection, None
            self.next_text = self.generate_text()
            delay = float(len(self.next_text) / self.CHAR_RATE) * 1000 + self.EPSILON
            if self.last_message_timestamp + datetime.timedelta(days=0, seconds=0,
                                                                milliseconds=delay) > datetime.datetime.now():
                self.selection = None
                return None, None
            else:
                ret_text = self.next_text
                self.next_text = None
                self.selection = None
                self.last_message_timestamp = datetime.datetime.now()
                return selection, ret_text.lower()

    def generate_text(self):
        if self.state == ChatState.CHAT:
            text = self.tagger.get_template(TemplateType.CHAT).text
        elif self.state == ChatState.ASK:
            text = self.tagger.get_template(TemplateType.ASK).text
        elif self.state == ChatState.TELL:
            template = self.tagger.get_template(TemplateType.TELL, self.template_type)
            text = self.fill_in_template(template)
        elif self.state == ChatState.SUGGEST:
            template = self.tagger.get_template(TemplateType.SUGGEST)
            text = self.fill_in_template(template)
        elif self.state == ChatState.ACCEPT:
            text = self.tagger.get_template(TemplateType.ACCEPT).text
        elif self.state == ChatState.REJECT:
            text = self.tagger.get_template(TemplateType.REJECT).text
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
                if len(self.mentioned_friends) == len(self.ranked_friends):
                    self.mentioned_friends = set()
                ctr = 0
                friend = self.ranked_friends[ctr]
                while friend in self.mentioned_friends and ctr < len(self.ranked_friends):
                    friend = self.ranked_friends[ctr]
                    ctr += 1
                self.mentioned_friends.add(friend)
                if entity == '<firstName>':
                    named_entities.append(friend.split()[0])
                elif entity == '<fullName>':
                    named_entities.append(friend)
        self.friend_ctr = ctr + 1
        return template.fill(entities, named_entities)

    def start_chat(self):
        return "hey"  # todo choose from template
