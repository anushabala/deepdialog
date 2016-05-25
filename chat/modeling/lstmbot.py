import random
import string
import operator
from chat.modeling.chatbot import ChatBotBase
from chat.nn import spec as specutil
from chat.nn import encdecspec
import sys

from collections import defaultdict
from chat.nn.encoderdecoder import EncoderDecoderModel
import chat.nn.vocabulary as vocabulary
from chat.nn.vocabulary import Vocabulary
import numpy as np
from chat.modeling.tagger import Entity
import datetime

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
    print "splitting", seqs
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


class LSTMChatBot(ChatBotBase):
    CHAR_RATE = 11
    SELECTION_DELAY = 1000
    EPSILON = 1500
    MAX_OUT_LEN = 50
    MENTION_WINDOW = 3

    def __init__(self, scenario, agent_num, tagger, model, include_features=True, name='LSTM'):
        self.scenario = scenario
        self.agent_num = agent_num
        self.friends = scenario["agents"][agent_num]["friends"]
        self.full_names_cased = {}
        self.first_names_to_full_names = {}
        self.probabilities = {}
        self.my_info = scenario["agents"][agent_num]["info"]
        r = np.random.random()
        self.my_turn = True if r < 0.5 else False
        print "My turn: ", self.my_turn
        self.tagger = tagger
        self.name = name
        self.model = model
        self._started = False
        self.include_features = include_features

        self.in_vocabulary = model.spec.in_vocabulary
        self.out_vocabulary = model.spec.out_vocabulary
        self.hidden_size = model.spec.hidden_size
        self.my_mentions = []
        self.partner_mentions = []
        self.h_t = self.model.spec.get_init_state().eval()
        # print self.h_t
        print "got initial hidden state"
        self.last_message_timestamp = datetime.datetime.now()
        self._ended = False
        self.partner_selected_connection = False
        self.partner_selection_message = None
        self.next_messages = []
        self.create_mappings()
        self.init_probabilities()
        print "finished initializing probabilities"

    def create_mappings(self):
        for friend in self.friends:
            name = friend["name"]
            self.full_names_cased[name.lower()] = name
            first_name = friend["name"].lower().split()[0]
            self.first_names_to_full_names[first_name] = name.lower()

    def rerank_friends(self, new_partner_mentions):
        if len(new_partner_mentions.keys()) == 0:
            return
        for entity_type in new_partner_mentions.keys():
            mentions = new_partner_mentions[entity_type]
            for friend in self.friends:
                name = friend["name"].lower()
                first_name = friend["name"].lower().split()[0]
                if entity_type == Entity.to_tag(Entity.FIRST_NAME):
                    if name in mentions:
                        self.probabilities[name] += 1
                    elif first_name in mentions:
                        self.probabilities[name] += 1
                elif entity_type == Entity.to_tag(Entity.MAJOR):
                    major = friend["school"]["major"].lower()
                    if major in mentions:
                        self.probabilities[name] += 1
                elif entity_type == Entity.to_tag(Entity.SCHOOL_NAME):
                    school = friend["school"]["name"].lower()
                    if school in mentions:
                        self.probabilities[name] += 1
                elif entity_type == Entity.to_tag(Entity.COMPANY_NAME):
                    company = friend["company"]["name"].lower()
                    if company in mentions:
                        self.probabilities[name] += 1

    def replace_entites_no_features(self, tagged_seq):
        print "LSTM tagged output:", tagged_seq
        # tagged_seq = tagged_seq.lower()
        tagged_seq = tagged_seq.replace(Vocabulary.END_OF_SENTENCE, "")
        tokens = tagged_seq.strip().split()
        new_sentence = []
        my_info = self.scenario["agents"][self.agent_num]
        partner_info = self.scenario["agents"][1-self.agent_num]
        new_mentions = defaultdict(list)

        my_mentions_flat = defaultdict(set)
        for mentions_dict in self.my_mentions:
            for entity_type in mentions_dict.keys():
                my_mentions_flat[entity_type].update(mentions_dict[entity_type])

        for token in tokens:
            tag, _ = get_tag_and_features(token)
            if tag is None:
                new_sentence.append(token)
                continue

            all_entities = get_all_entities_with_tag(tag, my_info)
            print token, tag
            if tag == Entity.to_tag(Entity.FIRST_NAME):
                sorted_probs = sorted(self.probabilities.items(), key=operator.itemgetter(1), reverse=True)
                sorted_choices = [a[0] for a in sorted_probs if a[0] not in my_mentions_flat[tag]]
                entity = sorted_choices[0]
            else:
                if tag == Entity.to_tag(Entity.SCHOOL_NAME):
                    entity = my_info["info"]["school"]["name"]
                elif tag == Entity.to_tag(Entity.MAJOR):
                    entity = my_info["info"]["school"]["major"]
                elif tag == Entity.to_tag(Entity.COMPANY_NAME):
                    entity = my_info["info"]["company"]["name"]

                if entity in my_mentions_flat[tag]:
                    choices = [c for c in all_entities if c not in my_mentions_flat[tag]]
                    if len(choices) > 0:
                        entity = np.random.choice(choices)
                    else:
                        entity = np.random.choice(all_entities)

            my_mentions_flat[tag].add(entity)
            new_sentence.append(entity)
            new_mentions[tag].append(entity)
            print "New sentence:", new_sentence
        return " ".join(new_sentence), new_mentions

    def replace_entities(self, tagged_seq):
        if not self.include_features:
            return self.replace_entites_no_features(tagged_seq)
        print "LSTM tagged output:", tagged_seq
        # tagged_seq = tagged_seq.lower()
        tagged_seq = tagged_seq.replace(Vocabulary.END_OF_SENTENCE, "")
        tokens = tagged_seq.strip().split()
        new_sentence = []
        my_info = self.scenario["agents"][self.agent_num]
        partner_info = self.scenario["agents"][1-self.agent_num]
        new_mentions = defaultdict(list)

        my_mentions_flat = defaultdict(set)
        for mentions_dict in self.my_mentions:
            for entity_type in mentions_dict.keys():
                my_mentions_flat[entity_type].update(mentions_dict[entity_type])

        partner_mentions_flat = defaultdict(set)
        for mentions_dict in self.partner_mentions:
            for entity_type in mentions_dict.keys():
                partner_mentions_flat[entity_type].update(mentions_dict[entity_type])

        for token in tokens:
            tag, features = get_tag_and_features(token)
            if tag is None:
                new_sentence.append(token)
                continue

            if features is None:
                # this should never happen? the bot should never generate something that doesn't match its information
                # todo default to some random behavior here, pass for now
                continue

            print "my mentions flat", my_mentions_flat
            print "partner mentions flat", partner_mentions_flat
            choices = []
            entity = None
            try:
                if "F:MENTIONED_BY_FRIEND" in features and "F:MENTIONED_BY_ME" in features:
                    choices = [c for c in my_mentions_flat[tag] if c in partner_mentions_flat[tag]]
                    assert len(choices) > 0
                    # if len(my_mentions_flat[tag]) > 0:
                    #     choices = []
                    # pass
                elif "F:MENTIONED_BY_FRIEND" in features and "F:MENTIONED_BY_ME" not in features:
                    choices = [c for c in partner_mentions_flat[tag] if c not in my_mentions_flat[tag]]
                    assert len(choices) > 0
                    # this should never happen, todo again default
                    # pass
                elif "F:MENTIONED_BY_ME" in features and "F:MENTIONED_BY_FRIEND" not in features:
                    choices = [c for c in my_mentions_flat[tag] if c not in partner_mentions_flat[tag]]
                    assert len(choices) > 0
                    # this should never happen, todo again default
                else:
                    # no mentions at all
                    all_entities = get_all_entities_with_tag(tag, my_info)
                    print all_entities, tag
                    choices = [c for c in all_entities if c not in my_mentions_flat[tag] and c not in partner_mentions_flat[tag]]
                    assert len(choices) > 0
            except AssertionError:
                # print choices
                # print tag, features
                # print self.my_mentions
                # print my_mentions_flat[tag]
                # print self.partner_mentions
                # print partner_mentions_flat[tag]
                # print "Mention tag mismatch; ignoring mention tags entirely"
                all_entities = get_all_entities_with_tag(tag, my_info)
                choices = all_entities
                print choices
            print "choices after mentions:", choices
            if tag == Entity.to_tag(Entity.FIRST_NAME):
                if "F:UNKNOWN" in features:
                    try:
                        assert "F:MENTIONED_BY_FRIEND" in features and len(choices) > 0
                    except AssertionError:
                        # todo do something here, can't generate unknown friend without previous mentions
                        pass
                    partner_friends = get_all_entities_with_tag(tag, partner_info)
                    choices = [c for c in partner_friends if c in choices]
                    entity = np.random.choice(choices)
                elif "F:KNOWN" in features:
                    sorted_probs = sorted(self.probabilities.items(), key=operator.itemgetter(1), reverse=True)
                    print sorted_probs
                    print choices
                    sorted_choices = [a[0] for a in sorted_probs if a[0] in choices]
                    entity = sorted_choices[0]
                else:
                    # todo maybe raise an error! must always generate either known or unknown friend
                    sorted_probs = sorted(self.probabilities.items(), key=operator.itemgetter(1), reverse=True)
                    sorted_choices = [a[0] for a in sorted_probs if a in choices]
                    entity = sorted_choices[0]
            else:
                if "F:MATCH_ME" in features:
                    if tag == Entity.to_tag(Entity.SCHOOL_NAME):
                        entity = my_info["info"]["school"]["name"]
                    elif tag == Entity.to_tag(Entity.MAJOR):
                        entity = my_info["info"]["school"]["major"]
                    elif tag == Entity.to_tag(Entity.COMPANY_NAME):
                        entity = my_info["info"]["company"]["name"]
                elif "F:MATCH_FRIEND" in features:
                    all_entities = get_all_entities_with_tag(tag, my_info)
                    if tag == Entity.to_tag(Entity.SCHOOL_NAME):
                        my_entity = my_info["info"]["school"]["name"]
                    elif tag == Entity.to_tag(Entity.MAJOR):
                        my_entity = my_info["info"]["school"]["major"]
                    else:
                        my_entity = my_info["info"]["company"]["name"]
                    print "MATCH FRIEND FEATURE: ", all_entities
                    choices = [c for c in all_entities if c in choices and c != my_entity]
                    print choices
                    entity = np.random.choice(choices)
                else:
                    # must be something that's been mentioned
                    try:
                        assert "F:MENTIONED_BY_FRIEND" in features and len(choices) > 0
                    except AssertionError:
                        # todo do something default
                        pass
                    all_entities = get_all_entities_with_tag(tag, my_info)
                    choices = [c for c in choices if c not in all_entities]
                    entity = np.random.choice(choices)

            my_mentions_flat[tag].add(entity)
            new_sentence.append(entity)
            new_mentions[tag].append(entity)
            print "New sentence:", new_sentence

        return " ".join(new_sentence), new_mentions

    def send(self):
        if self._ended:
            return None, None

        if not self.my_turn:
            # just send earlier messages if any
            if len(self.next_messages) == 0:
                return None, None
            next_message = self.next_messages[0]
            if SELECT in next_message:
                delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
                if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
                    return None, None
                else:
                    ret_text = self.next_messages.pop(0)
                    selection = ret_text.replace(SELECT, "").strip()
                    if selection in self.full_names_cased.keys():
                        selection = self.full_names_cased[selection]
                    return selection, None
            else:
                delay = float(len(next_message)) / self.CHAR_RATE * 1000 + random.uniform(0, self.EPSILON)
                if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
                    return None, None
                else:
                    ret_text = self.next_messages.pop(0).lower()
                    return None, ret_text

        if self.my_turn:
            pred_inds = []

            for i in range(self.MAX_OUT_LEN):
                write_dist = self.model._decoder_write(self.h_t)
                y_t = np.argmax(write_dist)
                p_y_t = write_dist[y_t] # probs for printing, if needed
                pred_inds.append(y_t)
                h_t = self.model._decoder_step(y_t, self.h_t)
                self.h_t = h_t
                if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
                    break

            y_words = self.out_vocabulary.indices_to_sentence(pred_inds)
            y_with_entities, new_mentions = self.replace_entities(y_words)
            self.update_mentions(new_mentions, mine=True)
            messages = split_into_utterances(y_with_entities)
            if messages[0] == '':
                messages.pop(0)

            # OVERRIDE LSTM OUTPUT IF PARTNER SELECTION MADE BUT BOT DOESN'T GENERATE SELECTION CORRECTLY
            self.next_messages.extend(messages)
            if self.partner_selected_connection and self.partner_selection_message not in self.next_messages:
                self.next_messages = []
                self.next_messages.append(self.partner_selection_message)

            self.my_turn = False

            next_message = self.next_messages[0]
            if SELECT in next_message:
                delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
                if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
                    return None, None
                else:
                    ret_text = self.next_messages.pop(0)
                    selection = ret_text.replace(SELECT, "").strip()
                    if selection in self.full_names_cased.keys():
                        selection = self.full_names_cased[selection]
                    return selection, None
            else:
                delay = float(len(next_message)) / self.CHAR_RATE * 1000 + random.uniform(0, self.EPSILON)
                if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
                    return None, None
                else:
                    ret_text = self.next_messages.pop(0).lower()
                    return None, ret_text

    def replace_with_tags(self, seq, found_entities, possible_entities, features):
        seq = seq.strip().lower()
        seq = str(seq).translate(string.maketrans("",""), string.punctuation)
        sentence = seq.split()
        new_sentence = []

        all_entities = [found_entities, possible_entities]
        all_matched_tokens = []

        for entity_dict in all_entities:
            for entity_type in entity_dict.keys():
                all_matched_tokens.extend(entity_dict[entity_type])

        print found_entities
        print possible_entities
        new_mentions = defaultdict(list)
        for entity_dict in all_entities:
            for entity_type in entity_dict.keys():
                for mentioned in entity_dict[entity_type]:
                    if mentioned in self.tagger.synonyms[entity_type].keys() and len(self.tagger.synonyms[entity_type][mentioned]) > 0:
                        new_mentions[Entity.to_tag(entity_type)].extend(self.tagger.synonyms[entity_type][mentioned])
                    else:
                        if mentioned in self.first_names_to_full_names.keys():
                            new_mentions[Entity.to_tag(entity_type)].append(self.first_names_to_full_names[mentioned])
                        else:
                            new_mentions[Entity.to_tag(entity_type)].append(mentioned)

        print new_mentions
        friend_mentions_flat = []
        for d in self.partner_mentions:
            for entity_type in d.keys():
                friend_mentions_flat.extend(d[entity_type])
        print friend_mentions_flat
        my_mentions_flat = []
        for d in self.my_mentions:
            for entity_type in d.keys():
                my_mentions_flat.extend(d[entity_type])

        for mentioned in friend_mentions_flat:
            if mentioned in all_matched_tokens:
                if "_<F:MENTIONED_BY_FRIEND>" not in features[mentioned]:
                    features[mentioned] += "_<F:MENTIONED_BY_FRIEND>"
            elif " " in mentioned:
                split = mentioned.split()
                for part in split:
                    if part in all_matched_tokens:
                        if "_<F:MENTIONED_BY_FRIEND>" not in features[part]:
                            features[part] += "_<F:MENTIONED_BY_FRIEND>"
            for token in all_matched_tokens:
                token = token.lower()
                if " " in token:
                    split = token.split()

                    for part in split:
                        if part == mentioned:
                            if "_<F:MENTIONED_BY_FRIEND>" not in features[token]:
                                features[token] += "_<F:MENTIONED_BY_FRIEND>"

        for mentioned in my_mentions_flat:
            if mentioned in all_matched_tokens:
                if "_<F:MENTIONED_BY_ME>" not in features[mentioned]:
                    features[mentioned] += "_<F:MENTIONED_BY_ME>"
            elif " " in mentioned:
                split = mentioned.split()
                for part in split:
                    if part in all_matched_tokens:
                        if "_<F:MENTIONED_BY_ME>" not in features[part]:
                            features[part] += "_<F:MENTIONED_BY_ME>"
            for token in all_matched_tokens:
                token = token.lower()
                if " " in token:
                    split = token.split()
                    for part in split:
                        if part in mentioned:
                            if "_<F:MENTIONED_BY_ME>" not in features[token]:
                                features[token] += "_<F:MENTIONED_BY_ME>"

        # print sentence
        for entity_type in Entity.types():
            for entity_dict in all_entities:
                if entity_type not in entity_dict.keys():
                    continue
                matched_tokens = entity_dict[entity_type]
                # if entity_type == Entity.SCHOOL_NAME:
                #     print matched_tokens
                for token in matched_tokens:
                    i = 0
                    while i < len(sentence):
                        if " " in token:
                            split_token = [t.strip() for t in token.split()]
                            try:
                                sentence_tokens = [w.strip() for w in sentence[i:i+len(split_token)]]
                                if split_token == sentence_tokens:
                                    if self.include_features:
                                        new_sentence.append("%s" % features[" ".join(sentence_tokens)])
                                    else:
                                        new_sentence.append("<%s>" % Entity.to_tag(entity_type))
                                    # if "krone" in split_token:
                                    #     print "SENTENCE", sentence
                                    #     print split_token
                                    #     print sentence_tokens, i
                                    #     print new_sentence
                                    i += len(split_token)
                                else:
                                    new_sentence.append(sentence[i])
                                    i += 1
                            except IndexError:
                                new_sentence.append(sentence[i])
                                i+=1
                        elif token == sentence[i]:
                            if self.include_features:
                                new_sentence.append(features[token])
                            else:
                                new_sentence.append("<%s>" % Entity.to_tag(entity_type))
                            i+=1
                        else:
                            new_sentence.append(sentence[i])
                            i+=1
                    sentence = new_sentence
                    # print sentence
                    new_sentence = []

        sentence = " ".join(sentence)
        # print sentence
        # for entity_type in Entity.types():
        #     for entity_dict in all_entities:
        #         matched_tokens = entity_dict[entity_type]
        #         for token in matched_tokens:
        #             if " " in token:
        #                 sentence = sentence.replace(token, Entity.to_tag(entity_type))
        return sentence, new_mentions

    def update_mentions(self, new_mentions, mine=False):
        to_update = self.my_mentions if mine else self.partner_mentions
        print "Updating (mine=%s)" % str(mine)
        print "before:", to_update
        print "new:", new_mentions
        if len(to_update) >= self.MENTION_WINDOW:
            to_update.pop(0)
        to_update.append(new_mentions)
        print "after:", to_update

    def receive(self, message):
        if self._ended:
            return
        self.last_message_timestamp = datetime.datetime.now()
        selection = False
        if SELECT in message:
            selection = True
            message = message.replace(SELECT, "")

        print "Raw message received:", message
        found_entities, possible_entities, features = self.tagger.tag_sentence(message, include_features=True, scenario=self.scenario, agent_idx=self.agent_num)

        tagged_msg, new_mentions = self.replace_with_tags(message, found_entities, possible_entities, features)

        if not self._started:
            if START not in message:
                tagged_msg = START + " " + tagged_msg.strip()
            self._started = True

        if message != "" and not selection:
            tagged_msg = SAY_DELIM + tagged_msg.strip()
        elif selection:
            tagged_msg = SELECT + " " + tagged_msg.strip()

        print "Tagged message received: ", tagged_msg
        print "New mentions:", new_mentions
        self.rerank_friends(new_mentions)
        self.update_mentions(new_mentions)
        print "Friend mentions:", self.partner_mentions
        self.my_turn = True

        x_inds = self.in_vocabulary.sentence_to_indices(tagged_msg)
        print x_inds
        self.h_t = self.model._encode(x_inds, self.h_t)

        ret_data = {"probs": self._get_probability_string(),
                    "confident_tags": self._get_entities_string(found_entities),
                    "possible_tags": self._get_entities_string(possible_entities)}
        return ret_data

    def init_probabilities(self):
        for friend in self.friends:
            self.probabilities[friend["name"].lower()] = 0.0

    def partner_selection(self, selection):
        selection = selection.lower()
        selection_message = "%s %s" % (SELECT, selection.lower())
        self.receive(selection_message)
        if selection in self.probabilities.keys():
            self.partner_selected_connection = True
            self.partner_selection_message = selection_message

    def end_chat(self):
        self.my_turn = False
        self._ended = True

    def start(self):
        if self.my_turn:
            self.receive("")
        else:
            pass