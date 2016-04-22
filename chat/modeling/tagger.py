import string

__author__ = 'anushabala'
import json
import os
import re


entity_pattern = r'<w+>'


class Entity(object):
    FULL_NAME='<fullName>'
    FIRST_NAME='<firstName>'
    SCHOOL_NAME='<schoolName>'
    MAJOR='<major>'
    COMPANY_NAME='<companyName>'

    @classmethod
    def types(cls):
        return [cls.FULL_NAME, cls.FIRST_NAME, cls.SCHOOL_NAME, cls.MAJOR, cls.COMPANY_NAME]


class TemplateType(object):
    CHAT=1
    SUGGEST=2
    ASK=3
    TELL=4

    @classmethod
    def types(cls):
        return [cls.CHAT, cls.SUGGEST, cls.ASK, cls.TELL]

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


class EntityTagger(object):
    scenarios = []
    templates = {template_type:[] for template_type in TemplateType.types()}
    entities = {entity:set() for entity in Entity.types()}

    def __init__(self, scenarios_file, templates_dir, synonyms_file=None):
        self.load_scenarios(open(scenarios_file, 'r'))
        if synonyms_file:
            self.load_synonyms(synonyms_file)
        self.load_entities()
        self.load_templates(templates_dir)

    def load_synonyms(self, synonyms_file):
        raise NotImplementedError

    def load_scenarios(self, scenarios_file):
        self.scenarios = json.load(scenarios_file, encoding='utf-8')

    def load_entities(self):
        for scenario in self.scenarios:
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

                for friend in agent_info["friends"]:
                    self.entities[Entity.MAJOR].add(friend["school"]["major"].lower())
                    self.entities[Entity.SCHOOL_NAME].add(friend["school"]["name"].lower())
                    self.entities[Entity.COMPANY_NAME].add(friend["company"]["name"].lower())
                    name = friend["name"].lower()
                    self.entities[Entity.FULL_NAME].add(name)
                    self.entities[Entity.FIRST_NAME].add(name.split()[0])

    def tag_sentence(self, sentence):
        sentence = sentence.strip().lower()
        sentence_mod = sentence.translate(string.maketrans("",""), string.punctuation)
        sentence_mod = sentence_mod.split()
        found_entities = {}

        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in self.entities[Entity.FIRST_NAME]:
                try:
                    next_word = sentence_mod[i+1]
                    if "%s %s" % (word, next_word) in self.entities[Entity.FULL_NAME]:
                        found_entities[Entity.FULL_NAME] = "%s %s" % (word, next_word)
                    else:
                        found_entities[Entity.FIRST_NAME] = word
                except IndexError:
                    found_entities[Entity.FIRST_NAME] = word
            elif word in self.entities[Entity.MAJOR]:
                found_entities[Entity.MAJOR] = word
            elif word in self.entities[Entity.SCHOOL_NAME]:
                found_entities[Entity.SCHOOL_NAME] = word
            elif word in self.entities[Entity.COMPANY_NAME]:
                found_entities[Entity.COMPANY_NAME] = word

        return found_entities

    def load_templates(self, templates_dir):
        for filename in os.listdir(templates_dir):
            f = open(os.path.join(templates_dir, filename))
            t_type = TemplateType.ASK
            if "chat" in filename:
                t_type = TemplateType.CHAT
            elif "suggest" in filename:
                t_type = TemplateType.SUGGEST
            elif "tell" in filename:
                t_type = TemplateType.TELL
            for line in f.readlines():
                line = line.strip().split(',')
                template = Template(line[0], line[2], line[1])
                self.templates[t_type].append(template)

    def generate(self, type, subtype, entities=[]):
        pass