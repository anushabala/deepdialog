import string

__author__ = 'anushabala'
import json
import os
import re
from collections import defaultdict
import random
import csv

entity_pattern = r'<\w+>'
SPECIAL_WORDS = ['select', 'say', 'name', 'my']

class UnsupportedOperationError(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)


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
        if type == Entity.FIRST_NAME:
            return "<firstName>"
        if type == Entity.SCHOOL_NAME:
            return "<schoolName>"
        if type == Entity.MAJOR:
            return "<major>"
        if type == Entity.COMPANY_NAME:
            return "<companyName>"

    @classmethod
    def to_tag(cls, type):
        if type == Entity.FULL_NAME:
            return "FRIEND_FULL_NAME"
        if type == Entity.FIRST_NAME:
            return "FRIEND_FULL_NAME"
        if type == Entity.SCHOOL_NAME:
            return "MY_SCHOOL"
        if type == Entity.MAJOR:
            return "MY_MAJOR"
        if type == Entity.COMPANY_NAME:
            return "MY_COMPANY"

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
            # if word[:i] == 'penn':
            #     print prefixes
    return prefixes


def get_acronyms(entity):
    words = entity.split()
    if len(words) < 2:
        return []
    return ["".join([w[0] for w in words])]


def find_unique_words(entity, all_entities):
    mod_entity = str(entity).translate(string.maketrans("",""), string.punctuation)
    words = mod_entity.split()
    unique_words = []
    if len(words) == 1:
        return []
    for word in words:
        unique = True
        for other_entity in all_entities:
            if word in other_entity and other_entity != entity:
                unique = False
        if unique:
            if word != entity:
                unique_words.append(word)

    return unique_words


class EntityTagger(object):
    scenarios = []
    templates = {template_type:[] for template_type in TemplateType.types()}
    entities = {entity:set() for entity in Entity.types()}
    synonyms = {entity:defaultdict(list) for entity in Entity.types()}

    def __init__(self, scenarios, templates_dir=None):
        self.scenarios = scenarios
        self.load_entities()
        self.compute_synonyms()
        if templates_dir:
            self.tag_only = False
            self.load_templates(templates_dir)
        else:
            self.tag_only = True

    def compute_synonyms(self):
        for entity_type in self.synonyms.keys():
            syn_dict = self.synonyms[entity_type]
            if entity_type == Entity.FULL_NAME or entity_type == Entity.FIRST_NAME:
                continue
            for entity in self.entities[entity_type]:

                entity_synonyms = []
                entity_synonyms.extend(get_prefixes(entity.lower()))
                entity_synonyms.extend(get_acronyms(entity.lower()))
                entity_synonyms.extend(find_unique_words(entity.lower(), self.entities[entity_type]))
                if entity_type == Entity.SCHOOL_NAME:
                    print entity_synonyms
                for syn in entity_synonyms:
                    syn_dict[syn].append(entity)

    def load_scenarios(self, scenarios_file):
        self.scenarios = json.load(scenarios_file, encoding='utf-8')

    def load_entities(self):
        for (key, scenario) in self.scenarios.iteritems():
            connection = scenario["connection"]["info"]
            self.entities[Entity.MAJOR].add(connection["school"]["major"].lower())
            school_name = connection["school"]["name"].lower()
            school_name = str(school_name).translate(string.maketrans("",""), string.punctuation).replace("  ", " ")
            self.entities[Entity.SCHOOL_NAME].add(school_name)
            self.entities[Entity.COMPANY_NAME].add(connection["company"]["name"].lower())
            name = connection["name"].lower()
            self.entities[Entity.FULL_NAME].add(name)
            self.entities[Entity.FIRST_NAME].add(name.split()[0])

            for agent in scenario["agents"]:
                agent_info = agent["info"]
                self.entities[Entity.MAJOR].add(agent_info["school"]["major"].lower())
                school_name = agent_info["school"]["name"].lower()
                school_name = str(school_name).translate(string.maketrans("",""), string.punctuation).replace("  ", " ")
                self.entities[Entity.SCHOOL_NAME].add(school_name)
                self.entities[Entity.SCHOOL_NAME].add(school_name)
                self.entities[Entity.COMPANY_NAME].add(agent_info["company"]["name"].lower())
                name = connection["name"].lower()
                self.entities[Entity.FULL_NAME].add(name)
                self.entities[Entity.FIRST_NAME].add(name.split()[0])

                for friend in agent["friends"]:
                    self.entities[Entity.MAJOR].add(friend["school"]["major"].lower())
                    school_name = friend["school"]["name"].lower()
                    school_name = str(school_name).translate(string.maketrans("",""), string.punctuation).replace("  ", " ")
                    self.entities[Entity.SCHOOL_NAME].add(school_name)
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
                    possible_matches[entity_type].append(prefix)
                    #todo somehow pass the actual entity name back to chatbot!! but the synonym itself is needed for
                    # sentence tagging so don't change this
        return possible_matches

    def ensure_unique(self, found_entities, possible_entities):
        unique_possible_matches = defaultdict(list)
        for entity_type in Entity.types():
            entities = found_entities[entity_type]
            for synonym in possible_entities[entity_type]:
                matches = self.synonyms[entity_type][synonym]
                not_subset = False
                for entity in matches:
                    if entity not in entities:
                        not_subset = True
                # if synonym == 'penn':
                #     print matches
                #     print not_subset
                if not_subset:
                    unique_possible_matches[entity_type].append(synonym)

        return unique_possible_matches

    def tag_sentence(self, sentence):
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
                        if word in SPECIAL_WORDS:
                            continue
                        found_entities[Entity.FIRST_NAME].append(word)
                except IndexError:
                    found_entities[Entity.FIRST_NAME].append(word)
            elif word in self.entities[Entity.MAJOR] and word not in SPECIAL_WORDS:
                found_entities[Entity.MAJOR].append(word)
            elif word in self.entities[Entity.SCHOOL_NAME] and word not in SPECIAL_WORDS:
                found_entities[Entity.SCHOOL_NAME].append(word)
            elif word in self.entities[Entity.COMPANY_NAME] and word not in SPECIAL_WORDS:
                found_entities[Entity.COMPANY_NAME].append(word)

        # try bi and tri grams
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in SPECIAL_WORDS:
                continue
            # if "nena" in sentence_mod:
            #     print sentence_mod, i
            if i+2 <= len(sentence_mod):
                bigram = " ".join(sentence_mod[i:i+2])
                if bigram in self.entities[Entity.MAJOR]:
                    found_entities[Entity.MAJOR].append(bigram)
                elif bigram in self.entities[Entity.SCHOOL_NAME]:
                    found_entities[Entity.SCHOOL_NAME].append(bigram)
                elif bigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(bigram)
                # if "nena" in bigram:
                #     print sentence_mod, '"',bigram,'"', found_entities
            if i+3 <= len(sentence_mod):
                trigram = " ".join(sentence_mod[i:i+3])
                if trigram in self.entities[Entity.MAJOR]:
                    found_entities[Entity.MAJOR].append(trigram)
                elif trigram in self.entities[Entity.SCHOOL_NAME]:

                    found_entities[Entity.SCHOOL_NAME].append(trigram)
                elif trigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(trigram)

            if i+4 <= len(sentence_mod):
                trigram = " ".join(sentence_mod[i:i+4])
                if trigram in self.entities[Entity.MAJOR]:
                    found_entities[Entity.MAJOR].append(trigram)
                elif trigram in self.entities[Entity.SCHOOL_NAME]:
                    found_entities[Entity.SCHOOL_NAME].append(trigram)
                elif trigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(trigram)
        # check if word matches any possible synonym
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in SPECIAL_WORDS:
                continue
            for entity_type in self.synonyms.keys():
                if len(self.synonyms[entity_type][word]) > 0:
                    # print entity_type, word, self.synonyms[entity_type][word]
                    possible_entities[entity_type].append(word)
                # if word == 'penn':
                    # print self.synonyms[entity_type][word]
                    # print possible_entities[entity_type]
        # print "possible", possible_entities

        # # check if prefix of any word matches a possible prefix in the dict
        # for i in range(0, len(sentence_mod)):
        #     word = sentence_mod[i]
        #     if word in SPECIAL_WORDS:
        #         continue
        #     possible_matches = self.possible_prefix_matches(word)
        #     for entity_type in possible_matches.keys():
        #         possible_entities[entity_type].extend(possible_matches[entity_type])

        possible_entities = self.ensure_unique(found_entities, possible_entities)
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
        if self.tag_only:
            raise UnsupportedOperationError("Template selection not supported: EntityTagger instance was defined "
                                            "in tagger-only mode.")
        templates = self.templates[type]
        if subtype is not None:
            templates = []
            for template in self.templates[type]:
                if template.type == subtype:
                    templates.append(template)
        return random.choice(templates)
