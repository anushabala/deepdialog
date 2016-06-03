import random
from chat.modeling.chatbot import ChatBotBase
from chat.modeling.dialogue_tracker import DialogueTracker
from chat.nn import encdecspec
import sys

from chat.nn.neural import NeuralBox
import chat.nn.vocabulary as vocabulary
import datetime
from dialogue_tracker import utterance_to_tokens
from chat.modeling import tokens as special_tokens

sys.modules['encdecspec'] = encdecspec
sys.modules['vocabulary'] = vocabulary


__author__ = 'anushabala'

START = "START"
SELECT = "SELECT NAME"
SAY_DELIM = "SAY "


def get_tag_and_features(token):
    if "<" not in token and ">" not in token:
        # not a tagged entity, return nonetypes
        return None, None

    token = token.split(">_<")
    assert len(token) > 0

    tag = token[0].strip("<>")
    if len(token) == 1:
        return tag, None

    features = [t.strip("<>") for t in token[1:]]
    return tag, features

def get_all_entities_with_tag(tag, json_info):
    if tag == Entity.to_tag(Entity.FULL_NAME):
        return list({friend["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.COMPANY_NAME):
        return list({friend["company"]["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.SCHOOL_NAME):
        return list({friend["school"]["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.MAJOR):
        return list({friend["school"]["major"].lower() for friend in json_info["friends"]})


def split_into_utterances(pred_seq):
    seqs = pred_seq.split(SAY_DELIM)
    if seqs[0].strip() == START and len(seqs) > 1:
        seqs.pop(0)
    new_seqs = []
    for s in seqs:
        if SELECT in s:
            i = s.index(SELECT)
            new_seqs.append(s[:i])
            new_seqs.append(s[i:])
        else:
            new_seqs.append(s)
    return new_seqs


class ModelBot(ChatBotBase):
    CHAR_RATE = 10.5
    SELECTION_DELAY = 1000
    EPSILON = 1500
    MAX_OUT_LEN = 50
    MENTION_WINDOW = 3

    def __init__(self, scenario, agent_num, box, lexicon, name='LSTM', args=None):
        # self.box = NeuralBox(model)
        self.box = box
        self.args = args
        self.tracker = DialogueTracker(lexicon, scenario, agent_num, args, self.box, None)
        self.name = name
        self.agent_num = agent_num
        self.scenario = scenario
        self.last_message_timestamp = datetime.datetime.now()
        # self.my_turn = True if np.random.random() >= 0.5 else False
        self.my_turn = True
        self._ended = False
        self.selection = None
        self.next_message = None
        self.partner_selected_connection = False
        self.partner_selection_message = None

        self.friends = scenario["agents"][agent_num]["friends"]
        self.full_names_cased = {}
        self.friend_names = []
        self.create_mappings()

    # todo do we really need this?
    def create_mappings(self):
        for friend in self.friends:
            name = friend["name"]
            self.full_names_cased[name.lower()] = name
            self.friend_names.append(name)

    def send_selection_if_possible(self):
        delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
        if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
            return None, None
        else:
            self.last_message_timestamp = datetime.datetime.now()
            selection = self.selection
            self.selection = None
            return selection, None

    def send_message_if_possible(self):
        message = self.next_message
        delay = float(len(message)) / self.CHAR_RATE * 1000 + random.uniform(0, self.EPSILON)
        if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
            return None, None
        else:
            self.last_message_timestamp = datetime.datetime.now()
            self.next_message = None
            return None, message

    def send(self):
        if self._ended:
            return None, None

        if not self.my_turn:

            return None, None

        # first check if previous messages need to be sent first

        if self.selection is not None:
           return self.send_selection_if_possible()

        if self.next_message is not None:
            return self.send_message_if_possible()

        tokens, end_turn = self.tracker.generate_add(self.agent_num)
        self.my_turn = False if end_turn else True
        if tokens[0] == special_tokens.SELECT_NAME:
            name = " ".join(tokens[1:])
            if name in self.full_names_cased.keys() or name in self.friend_names:
                selection = self.full_names_cased[name]
                self.selection = selection
                return self.send_selection_if_possible()
            else:
                return None, None # invalid selection?

        # Override LSTM output if partner selects connection
        if self.partner_selected_connection:
            self.next_message = self.partner_selection_message
        else:
            message = " ".join(tokens)
            self.next_message = message

        return self.send_message_if_possible()

    def receive(self, message):
        if self._ended:
            return
        self.last_message_timestamp = datetime.datetime.now()
        raw_tokens = utterance_to_tokens(message)
        self.tracker.parse_add(1 - self.agent_num, raw_tokens, True)
        self.my_turn = True

    def partner_selection(self, selection):
        selection = selection.lower()
        selection_message = "%s %s" % (special_tokens.SELECT_NAME, selection.lower())
        self.receive(selection_message)
        if selection in self.probabilities.keys():
            self.partner_selected_connection = True
            self.partner_selection_message = selection_message

    def end_chat(self):
        self.my_turn = False
        self._ended = True

    def start(self):
        pass