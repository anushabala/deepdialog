import string

__author__ = 'anushabala'
import json
import os
import re
from collections import defaultdict
import random
import csv

entity_pattern = r'<\w+>'


class Entity(object):
    FULL_NAME='<fullName>'
    FIRST_NAME='<firstName>'
    SCHOOL_NAME='<schoolName>'
    MAJOR='<major>'
    COMPANY_NAME='<companyName>'

    @classmethod
    def types(cls):
        return [cls.FULL_NAME, cls.FIRST_NAME, cls.SCHOOL_NAME, cls.MAJOR, cls.COMPANY_NAME]

    @classmethod
    def to_str(cls, type):
        if type == Entity.FULL_NAME:
            return "<fullName>"
        if type == Entity.FULL_NAME:
            return "<firstName>"
        if type == Entity.SCHOOL_NAME:
            return "<schoolName>"
        if type == Entity.MAJOR:
            return "<major>"
        if type == Entity.COMPANY_NAME:
            return "<companyName>"


class TemplateType(object):
    CHAT=1
    SUGGEST=2
    ASK=3
    TELL=4
    ACCEPT=5
    REJECT=6

    @classmethod
    def types(cls):
        return [cls.CHAT, cls.SUGGEST, cls.ASK, cls.TELL, cls.ACCEPT, cls.REJECT]

    @classmethod
    def subtypes(cls, type):
        if type == cls.CHAT:
            return ["CHAT"]
        elif type == cls.SUGGEST:
            return ["SUGGEST"]
        elif type == cls.ASK:
            return ["ASK_ALL", "ASK_SCHOOL_MAJOR","ASK_COMPANY","ASK_SCHOOL","ASK_MAJOR"]
        elif type == cls.TELL:
            return ["TELL_ALL", "TELL_SCHOOL_MAJOR", "TELL_COMPANY"]


class Template(object):
    num_entities = 0
    type = None
    text = None

    def __init__(self, text, num_entities, type):
        self.text = text
        self.num_entities = num_entities
        self.type = type

    def get_entities(self):
        if self.num_entities == 0:
            return []
        return re.findall(entity_pattern, self.text)

    def fill(self, matched_entities, named_entities):
        out = self.text
        for i in range(0, len(named_entities)):
            entity = matched_entities[i]
            out = out.replace(entity, named_entities[i], 1)
        return out


def get_prefixes(entity, min_length=3, max_length=5):
    prefixes = []
    words = entity.split()
    for word in words:
        if len(word) < max_length:
            continue
        for i in range(min_length, max_length):
            prefixes.append(word[:i])
    return prefixes


def get_acronyms(entity):
    words = entity.split()
    if len(words) < 2:
        return []
    return ["".join([w[0] for w in words])]


def find_unique_words(entity, all_entities):
    words = entity.split()
    unique_words = []
    if len(words) == 1:
        return []
    for word in words:
        unique = True
        for other_entity in all_entities:
            if word in other_entity:
                unique = False
        if unique:
            if word != entity:
                unique_words.append(word)

    return []


class EntityTagger(object):
    scenarios = []
    templates = {template_type:[] for template_type in TemplateType.types()}
    entities = {entity:set() for entity in Entity.types()}
    synonyms = {entity:defaultdict(list) for entity in Entity.types()}

    def __init__(self, scenarios, templates_dir):
        self.scenarios = scenarios
        self.load_entities()
        self.compute_synonyms()
        self.load_templates(templates_dir)

    def compute_synonyms(self):
        print "computing synonyms"
        print self.synonyms.keys()
        for entity_type in self.synonyms.keys():
            syn_dict = self.synonyms[entity_type]
            if entity_type == Entity.FULL_NAME or entity_type == Entity.FIRST_NAME:
                continue
            for entity in self.entities[entity_type]:

                entity_synonyms = []
                entity_synonyms.extend(get_prefixes(entity))
                entity_synonyms.extend(get_acronyms(entity))
                entity_synonyms.extend(find_unique_words(entity, self.entities[entity_type]))
                for syn in entity_synonyms:
                    syn_dict[syn].append(entity)



    def load_scenarios(self, scenarios_file):
        self.scenarios = json.load(scenarios_file, encoding='utf-8')

    def load_entities(self):
        for (key, scenario) in self.scenarios.iteritems():
            connection = scenario["connection"]["info"]
            self.entities[Entity.MAJOR].add(connection["school"]["major"].lower())
            self.entities[Entity.SCHOOL_NAME].add(connection["school"]["name"].lower())
            self.entities[Entity.COMPANY_NAME].add(connection["company"]["name"].lower())
            name = connection["name"].lower()
            self.entities[Entity.FULL_NAME].add(name)
            self.entities[Entity.FIRST_NAME].add(name.split()[0])

            for agent in scenario["agents"]:
                agent_info = agent["info"]
                self.entities[Entity.MAJOR].add(agent_info["school"]["major"].lower())
                self.entities[Entity.SCHOOL_NAME].add(agent_info["school"]["name"].lower())
                self.entities[Entity.COMPANY_NAME].add(agent_info["company"]["name"].lower())
                name = connection["name"].lower()
                self.entities[Entity.FULL_NAME].add(name)
                self.entities[Entity.FIRST_NAME].add(name.split()[0])

                for friend in agent["friends"]:
                    self.entities[Entity.MAJOR].add(friend["school"]["major"].lower())
                    self.entities[Entity.SCHOOL_NAME].add(friend["school"]["name"].lower())
                    self.entities[Entity.COMPANY_NAME].add(friend["company"]["name"].lower())
                    name = friend["name"].lower()
                    self.entities[Entity.FULL_NAME].add(name)
                    self.entities[Entity.FIRST_NAME].add(name.split()[0])

    def possible_prefix_matches(self, word):
        possible_matches = defaultdict(list)
        prefixes = get_prefixes(word)
        for prefix in prefixes:
            for entity_type in Entity.types():
                if entity_type == Entity.FIRST_NAME or entity_type == Entity.FULL_NAME:
                    continue
                if prefix in self.synonyms[entity_type].keys():
                    possible_matches[entity_type].extend(self.synonyms[entity_type][prefix])
        return possible_matches

    def tag_sentence(self, sentence):
        # todo do prefix match and return confidences
        # todo edit distances
        print sentence
        sentence = sentence.strip().lower()
        sentence_mod = sentence.translate(string.maketrans("",""), string.punctuation)
        sentence_mod = sentence_mod.split()
        found_entities = defaultdict(list)
        possible_entities = defaultdict(list)
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in self.entities[Entity.FIRST_NAME]:
                try:
                    next_word = sentence_mod[i+1]
                    if "%s %s" % (word, next_word) in self.entities[Entity.FULL_NAME]:
                        found_entities[Entity.FULL_NAME].append("%s %s" % (word, next_word))
                    else:
                        found_entities[Entity.FIRST_NAME].append(word)
                except IndexError:
                    found_entities[Entity.FIRST_NAME].append(word)
            elif word in self.entities[Entity.MAJOR]:
                found_entities[Entity.MAJOR].append(word)
            elif word in self.entities[Entity.SCHOOL_NAME]:
                found_entities[Entity.SCHOOL_NAME].append(word)
            elif word in self.entities[Entity.COMPANY_NAME]:
                found_entities[Entity.COMPANY_NAME].append(word)

        # # try bi and tri grams
        # for i in range(0, len(sentence_mod)):
        #     if i+2 <= len(sentence_mod):
        #         bigram = " ".join(sentence_mod[i:i+2])
        #         if bigram in self.entities[Entity.MAJOR]:
        #             found_entities[Entity.MAJOR].append(word)
        #         elif bigram in self.entities[Entity.SCHOOL_NAME]:
        #             found_entities[Entity.SCHOOL_NAME].append(word)
        #         elif bigram in self.entities[Entity.COMPANY_NAME]:
        #             found_entities[Entity.COMPANY_NAME].append(word)
        #     if i+3 <= len(sentence_mod):
        #         trigram = " ".join(sentence_mod[i:i+2])
        #         if trigram in self.entities[Entity.MAJOR]:
        #             found_entities[Entity.MAJOR].append(word)
        #         elif trigram in self.entities[Entity.SCHOOL_NAME]:
        #             found_entities[Entity.SCHOOL_NAME].append(word)
        #         elif trigram in self.entities[Entity.COMPANY_NAME]:
        #             found_entities[Entity.COMPANY_NAME].append(word)

        # check if word matches any possible synonym
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            for entity_type in self.synonyms.keys():
                possible_entities[entity_type].extend(self.synonyms[entity_type][word])


        # check if prefix of any word matches a possible prefix in the dict
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            possible_matches = self.possible_prefix_matches(word)
            for entity_type in possible_matches.keys():
                possible_entities[entity_type].extend(possible_matches[entity_type])

        print "[Tagger] Tagged sentence %s\tEntities:" % sentence, found_entities
        print "[Tagger] Possible matches\t", possible_entities
        possible_entities = {key:set(entities) for (key, entities) in possible_entities.items()}
        return found_entities, possible_entities

    def load_templates(self, templates_dir):
        for filename in os.listdir(templates_dir):
            reader = csv.reader(open(os.path.join(templates_dir, filename)))
            t_type = TemplateType.ASK
            if "chat" in filename:
                t_type = TemplateType.CHAT
            elif "suggest" in filename:
                t_type = TemplateType.SUGGEST
            elif "tell" in filename:
                t_type = TemplateType.TELL
            elif "accept" in filename:
                t_type = TemplateType.ACCEPT
            elif "reject" in filename:
                t_type = TemplateType.REJECT
            for line in reader:
                if line[0].startswith('#') or len(line) < 3:
                    continue
                template = Template(line[0], line[2], line[1])
                self.templates[t_type].append(template)

    def get_template(self, type, subtype=None):
        templates = self.templates[type]
        if subtype is not None:
            templates = []
            for template in self.templates[type]:
                if template.type == subtype:
                    templates.append(template)
        return random.choice(templates)
