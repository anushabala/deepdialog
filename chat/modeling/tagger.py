__author__ = 'anushabala'
import json
import os
import re


entity_pattern = r'<w+>'


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

    def __init__(self, scenarios_file, templates_dir, entities_file, synonyms_file=None):
        self.load_scenarios(open(scenarios_file, 'r'))
        if synonyms_file:
            self.load_synonyms(synonyms_file)

    def load_synonyms(self, synonyms_file):
        raise NotImplementedError

    def load_scenarios(self, scenarios_file):
        self.scenarios = json.load(scenarios_file, encoding='utf-8')

    def load_entities(self):

    def tag_sentence(self, scenario_id, agent_num, ):


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