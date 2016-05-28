import string
import sys

__author__ = 'anushabala'
import json
import os
import re
from collections import defaultdict
import random
import csv

entity_pattern = r'<\w+>'
SPECIAL_WORDS = ['select', 'say', 'name', 'my']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ' ']

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
            return "T:FRIEND"
        if type == Entity.FIRST_NAME:
            return "T:FRIEND"
        if type == Entity.SCHOOL_NAME:
            return "T:SCHOOL"
        if type == Entity.MAJOR:
            return "T:MAJOR"
        if type == Entity.COMPANY_NAME:
            return "T:COMPANY"

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
    acronyms = ["".join([w[0] for w in words])]
    if "of" in words:
        acronym = ''
        for w in words:
            acronym += w[0] if w != 'of' else ' '+w+' '
        acronym = acronym.strip()
        acronyms.append(acronym)

    return acronyms


def get_single_edits_for_word(entity):
    edits = set()
    edit_types = ['ins', 'del', 'sub', 'trans']
    for edit in edit_types:
        for i in range(0, len(entity)):
            prefix = entity[0:i]
            if edit == 'ins':
                suffix = entity[i:]
                for c in alphabet:
                    new_word = prefix + c + suffix
                    new_word = new_word.strip()
                    if new_word != entity:
                        edits.add(new_word)
            elif edit == 'del':
                suffix = entity[i+1:]
                new_word = prefix + suffix
                new_word = new_word.strip()
                edits.add(new_word)
            elif edit == 'sub':
                suffix = entity[i+1:]
                for c in alphabet:
                    if c != entity[i]:
                        new_word = prefix + c + suffix
                        new_word = new_word.strip()
                        if new_word != entity:
                            edits.add(new_word)
            elif edit == 'trans':
                for j in range(i+1, len(entity)):
                    mid = entity[i+1:j]
                    suffix = entity[j+1:]
                    new_word = prefix + entity[j] + mid + entity[i] + suffix
                    new_word = new_word.strip()
                    if new_word != entity:
                        edits.add(new_word)
    return edits


def get_entity_edits(entity, num_edits=1):
    prev_level = [entity]
    edits = set()
    for i in range(0, num_edits):
        for word in prev_level:
            edits.update(get_single_edits_for_word(word))
            edits = set([t for t in edits if t not in prev_level and t != entity])

        prev_level = set()
        prev_level.update(edits)
        edits = set()

    return prev_level


def find_unique_words(entity, all_entities):
    mod_entity = str(entity.strip()).translate(string.maketrans("",""), string.punctuation)
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
    edits = []
    for word in unique_words:
        edits.extend(get_entity_edits(word))

    unique_words.extend(edits)
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

            for entity in self.entities[entity_type]:

                entity_synonyms = []
                if entity_type != Entity.FULL_NAME and entity_type != Entity.FIRST_NAME:
                    entity_synonyms.extend(get_prefixes(entity.lower()))
                    entity_synonyms.extend(get_acronyms(entity.lower()))
                    entity_synonyms.extend(find_unique_words(entity.lower(), self.entities[entity_type]))
                    entity_synonyms.extend(get_entity_edits(entity, num_edits=1))
                else:
                    first_name = entity.split(" ")[0]
                    entity_synonyms.extend(get_entity_edits(first_name, num_edits=1))
                    entity_synonyms.extend(get_entity_edits(entity, num_edits=1))
                # if entity_type == Entity.SCHOOL_NAME:
                #     print entity_synonyms
                for syn in entity_synonyms:
                    syn_dict[syn].append(entity)

    def load_scenarios(self, scenarios_file):
        self.scenarios = json.load(scenarios_file)

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
        print prefixes
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
            # print "exact match entities found:", entity_type, entities
            for synonym in possible_entities[entity_type]:
                matches = self.synonyms[entity_type][synonym]
                # print "matches found for synonym:", entity_type, synonym, matches
                not_subset = False
                for entity in matches:
                    if entity not in entities:
                        not_subset = True
                # if synonym == 'penn':
                #     print matches
                #     print not_subset
                # print not_subset
                if not_subset:
                    unique_possible_matches[entity_type].append(synonym)

        return unique_possible_matches

    def get_features(self, entity, entity_type, scenario, agent_idx):
        tag = "<%s>" % Entity.to_tag(entity_type)
        if entity_type == Entity.FIRST_NAME:
            my_friends = scenario["agents"][agent_idx]["friends"]
            found = False
            for friend in my_friends:
                name = friend["name"].lower()
                if entity == name.split()[0]:
                    found = True
                    tag += "_<F:KNOWN>"
            if not found:
                tag += "_<F:UNKNOWN>"
        elif entity_type == Entity.FULL_NAME:
            my_friends = scenario["agents"][agent_idx]["friends"]
            found = False
            for friend in my_friends:
                name = friend["name"].lower()
                if entity == name:
                    found = True
                    tag += "_<F:KNOWN>"
            if not found:
                tag += "_<F:UNKNOWN>"
        elif entity_type == Entity.SCHOOL_NAME:
            my_school = scenario["agents"][agent_idx]["info"]["school"]["name"].lower()
            if entity == my_school:
                tag += "_<F:MATCH_ME>"
            my_friends = scenario["agents"][agent_idx]["friends"]
            for friend in my_friends:
                school = friend["school"]["name"].lower().replace(" - ", " ") # todo generalize to all punctuation
                if entity == school:
                    tag += "_<F:MATCH_FRIEND>"
                    break
        elif entity_type == Entity.COMPANY_NAME:
            my_company = scenario["agents"][agent_idx]["info"]["company"]["name"].lower()
            if entity == my_company:
                tag += "_<F:MATCH_ME>"
            my_friends = scenario["agents"][agent_idx]["friends"]
            for friend in my_friends:
                company = friend["company"]["name"].lower()
                if entity == company:
                    tag += "_<F:MATCH_FRIEND>"
                    break
        elif entity_type == Entity.MAJOR:
            my_major = scenario["agents"][agent_idx]["info"]["school"]["major"].lower()
            if entity == my_major:
                tag += "_<F:MATCH_ME>"
            my_friends = scenario["agents"][agent_idx]["friends"]
            for friend in my_friends:
                major = friend["school"]["major"].lower()
                if entity == major:
                    tag += "_<F:MATCH_FRIEND>"
                    break

        return tag

    def tag_sentence(self, sentence, include_features=False, scenario=None, agent_idx=-1):
        features = defaultdict(str)
        sentence = sentence.strip().lower()
        sentence_mod = str(sentence).translate(string.maketrans("",""), string.punctuation)
        sentence_mod = sentence_mod.split()
        found_entities = defaultdict(list)
        possible_entities = defaultdict(list)
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in self.entities[Entity.FIRST_NAME]:
                try:
                    next_word = sentence_mod[i+1]
                    if "%s %s" % (word, next_word) in self.entities[Entity.FULL_NAME]:
                        token = "%s %s" % (word, next_word)
                        found_entities[Entity.FULL_NAME].append(token)
                        if include_features:
                            f = self.get_features(token, Entity.FULL_NAME, scenario, agent_idx)
                            features[token] = f
                    else:
                        if word in SPECIAL_WORDS:
                            continue
                        found_entities[Entity.FIRST_NAME].append(word)
                        if include_features:
                            f = self.get_features(word, Entity.FIRST_NAME, scenario, agent_idx)
                            features[word] = f
                except IndexError:
                    found_entities[Entity.FIRST_NAME].append(word)
                    if include_features:
                        f = self.get_features(word, Entity.FIRST_NAME, scenario, agent_idx)
                        features[word] = f
            elif word in self.entities[Entity.MAJOR] and word not in SPECIAL_WORDS:
                found_entities[Entity.MAJOR].append(word)
                if include_features:
                    f = self.get_features(word, Entity.MAJOR, scenario, agent_idx)
                    features[word] = f
            elif word in self.entities[Entity.SCHOOL_NAME] and word not in SPECIAL_WORDS:
                found_entities[Entity.SCHOOL_NAME].append(word)
                if include_features:
                    f = self.get_features(word, Entity.SCHOOL_NAME, scenario, agent_idx)
                    features[word] = f
            elif word in self.entities[Entity.COMPANY_NAME] and word not in SPECIAL_WORDS:
                found_entities[Entity.COMPANY_NAME].append(word)
                if include_features:
                    f = self.get_features(word, Entity.COMPANY_NAME, scenario, agent_idx)
                    features[word] = f

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
                    if include_features:
                        f = self.get_features(word, Entity.MAJOR, scenario, agent_idx)
                        features[bigram] = f
                elif bigram in self.entities[Entity.SCHOOL_NAME]:
                    found_entities[Entity.SCHOOL_NAME].append(bigram)
                    if include_features:
                        f = self.get_features(word, Entity.SCHOOL_NAME, scenario, agent_idx)
                        features[bigram] = f
                elif bigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(bigram)
                    if include_features:
                        f = self.get_features(word, Entity.COMPANY_NAME, scenario, agent_idx)
                        features[bigram] = f
                else:
                    for entity_type in self.synonyms.keys():
                        if len(self.synonyms[entity_type][bigram]) > 0:
                            # print "Found %s in synonyms dictionary" % word
                            # print entity_type, word, self.synonyms[entity_type][word]

                            if include_features:
                                possible_entities[entity_type].append(bigram)
                                entity = self.synonyms[entity_type][bigram][0]
                                f = self.get_features(entity, entity_type, scenario, agent_idx)
                                features[bigram] = f
                            else:
                                # print self.synonyms[entity_type][word]
                                possible_entities[entity_type].append(bigram)
                # if "nena" in bigram:
                #     print sentence_mod, '"',bigram,'"', found_entities
            if i+3 <= len(sentence_mod):
                trigram = " ".join(sentence_mod[i:i+3])
                if trigram in self.entities[Entity.MAJOR]:
                    found_entities[Entity.MAJOR].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.MAJOR, scenario, agent_idx)
                        features[trigram] = f
                elif trigram in self.entities[Entity.SCHOOL_NAME]:
                    found_entities[Entity.SCHOOL_NAME].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.SCHOOL_NAME, scenario, agent_idx)
                        features[trigram] = f
                elif trigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.COMPANY_NAME, scenario, agent_idx)
                        features[trigram] = f
                else:
                    for entity_type in self.synonyms.keys():
                        if len(self.synonyms[entity_type][trigram]) > 0:
                            # print "Found %s in synonyms dictionary" % word
                            # print entity_type, word, self.synonyms[entity_type][word]

                            if include_features:
                                possible_entities[entity_type].append(trigram)
                                entity = self.synonyms[entity_type][trigram][0]
                                f = self.get_features(entity, entity_type, scenario, agent_idx)
                                features[trigram] = f
                            else:
                                # print self.synonyms[entity_type][word]
                                possible_entities[entity_type].append(trigram)

            if i+4 <= len(sentence_mod):
                trigram = " ".join(sentence_mod[i:i+4])
                if trigram in self.entities[Entity.MAJOR]:
                    found_entities[Entity.MAJOR].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.MAJOR, scenario, agent_idx)
                        features[trigram] = f
                elif trigram in self.entities[Entity.SCHOOL_NAME]:
                    found_entities[Entity.SCHOOL_NAME].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.SCHOOL_NAME, scenario, agent_idx)
                        features[trigram] = f
                elif trigram in self.entities[Entity.COMPANY_NAME]:
                    found_entities[Entity.COMPANY_NAME].append(trigram)
                    if include_features:
                        f = self.get_features(word, Entity.COMPANY_NAME, scenario, agent_idx)
                        features[trigram] = f
                else:
                    for entity_type in self.synonyms.keys():
                        if len(self.synonyms[entity_type][trigram]) > 0:
                            # print "Found %s in synonyms dictionary" % word
                            # print entity_type, word, self.synonyms[entity_type][word]

                            if include_features:
                                possible_entities[entity_type].append(trigram)
                                entity = self.synonyms[entity_type][trigram][0]
                                f = self.get_features(entity, entity_type, scenario, agent_idx)
                                features[trigram] = f
                            else:
                                # print self.synonyms[entity_type][word]
                                possible_entities[entity_type].append(trigram)
        # check if word matches any possible synonym
        for i in range(0, len(sentence_mod)):
            word = sentence_mod[i]
            if word in SPECIAL_WORDS:
                continue
            for entity_type in self.synonyms.keys():
                if len(self.synonyms[entity_type][word]) > 0:
                    # print "Found %s in synonyms dictionary" % word
                    # print entity_type, word, self.synonyms[entity_type][word]

                    if include_features:
                        possible_entities[entity_type].append(word)
                        entity = self.synonyms[entity_type][word][0]
                        f = self.get_features(entity, entity_type, scenario, agent_idx)
                        features[word] = f
                    else:
                        # print self.synonyms[entity_type][word]
                        possible_entities[entity_type].append(word)
                        # possible_entities[entity_type].extend(self.synonyms[entity_type][word])
                        # print "Found synonym for %s, %s" % (word, entity_type)
                        # print possible_entities[entity_type]
                        # print "Adding to possible entities", possible_entities
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

        # print "possible entities in tagger before ensure unique", possible_entities
        possible_entities = self.ensure_unique(found_entities, possible_entities)
        if not include_features:
            # replace all synonyms by entity
            new_possible_entities = defaultdict(list)
            for entity_type in possible_entities.keys():
                synonyms = possible_entities[entity_type]
                for syn in synonyms:
                    new_possible_entities[entity_type].extend(self.synonyms[entity_type][syn])
            possible_entities = new_possible_entities
        possible_entities = {key:set(entities) for (key, entities) in possible_entities.items()}
        # print "possible entities in tagger:", possible_entities
        if include_features:
            return found_entities, possible_entities, features
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
