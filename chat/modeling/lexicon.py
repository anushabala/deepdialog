from collections import defaultdict
from kb import KB

def get_prefixes(entity, min_length=3, max_length=5):
    # computer science => ['comp sci', ...]
    prefixes = []
    words = entity.split()
    candidates = ['']
    for word in words:
        new_candidates = []
        for c in candidates:
            if len(word) < max_length:  # Keep word
                new_candidates.append(c + ' ' + word)
            else:  # Shorten
                for i in range(min_length, max_length):
                    new_candidates.append(c + ' ' + word[:i])
        candidates = new_candidates
    return [c[1:] for c in candidates if c[1:] != entity]

def get_acronyms(entity):
    words = entity.split()
    if len(words) < 2:
        return []
    acronyms = [''.join([w[0] for w in words])]
    if 'of' in words:
        # handle 'u of p'
        acronym = ''
        for w in words:
            acronym += w[0] if w != 'of' else ' '+w+' '
        acronyms.append(acronym)
        # handle 'upenn'
        acronym = ''
        for w in words[:-1]:
            acronym += w[0] if w != 'of' else ''
        acronym += words[-1][:4]
        acronyms.append(acronym)

    return acronyms

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ' ']
def get_edits(entity):
    if len(entity) < 3:
        return []
    edits = []
    for i in range(len(entity) + 1):
        prefix = entity[:i]
        # Insert
        suffix = entity[i:]
        for c in alphabet:
            new_word = prefix + c + suffix
            edits.append(new_word)

        if i == len(entity):
            continue

        # Delete
        suffix = entity[i+1:]
        new_word = prefix + suffix
        edits.append(new_word)

        # Substitute
        suffix = entity[i+1:]
        for c in alphabet:
            if c != entity[i]:
                new_word = prefix + c + suffix
                edits.append(new_word)

        # Transposition
        for j in range(i+1, len(entity)):
            mid = entity[i+1:j]
            suffix = entity[j+1:]
            new_word = prefix + entity[j] + mid + entity[i] + suffix
            new_word = new_word.strip()
            if new_word != entity:
                edits.append(new_word)
    return edits


def get_morphological_variants(entity):
    # cooking => cook, cooker
    results = []
    for suffix in ['ing']:
        if entity.endswith(suffix):
            base = entity[:-len(suffix)]
            results.append(base)
            results.append(base + 's')
            results.append(base + 'er')
            results.append(base + 'ers')
    return results

############################################################

class Lexicon(object):
    '''
    A lexicon maps freeform phrases to canonicalized entity.
    The current lexicon just uses several heuristics to do the matching.
    '''
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.lexicon = defaultdict(list)  # Mapping from string -> list of (entity, type)
        self.load_entities()
        self.compute_synonyms()
        self.add_numbers()
        print 'Created lexicon: %d phrases mapping to %d entities' % (len(self.lexicon), len(self.entities))

    def load_entities(self):
        for scenario in self.scenarios.values():
            for agent in [0, 1]:
                self.kb = KB(scenario, agent)
                for row in self.kb.table:
                    for entity, type in row.values():
                        self._add_entity(type, entity)

    def _add_entity(self, type, entity):
        # Keep track of number of times words in this entity shows up
        if entity not in self.entities:
            for word in entity.split(' '):
                self.word_counts[word] += 1
        self.entities[entity] = type

    def lookup(self, phrase):
        return self.lexicon.get(phrase, [])

    def compute_synonyms(self):
        # Special cases
        for entity, type in self.entities.items():
            #print entity
            phrases = [entity]  # Representations of the canonical entity
            # Consider any word in the entity that's unique
            # Example: entity = 'university of california', 'university' would not be unique, but 'california' would be
            if ' ' in entity:
                for word in entity.split(' '):
                    if len(word) >= 3 and self.word_counts[word] == 1:
                        phrases.append(word)
            # Consider removing stop words
            mod_entity = entity
            for s in [' of ', ' - ']:
                mod_entity = mod_entity.replace(s, ' ')
            if entity != mod_entity:
                phrases.append(mod_entity)

            # Expand!
            synonyms = []
            # Special case
            if entity == 'facebook':
                synonyms.append('fb')
            if type == 'person':
                first_name = entity.split(' ')[0]
                if len(first_name) >= 3 and first_name not in synonyms:
                    synonyms.append(first_name)
            # General
            for phrase in phrases:
                synonyms.append(phrase)
                if type != 'person':
                    synonyms.extend(get_edits(phrase))
                    synonyms.extend(get_morphological_variants(phrase))
                    synonyms.extend(get_prefixes(phrase))
                    synonyms.extend(get_acronyms(phrase))

            # Add to lexicon
            for synonym in set(synonyms):
                #print synonym, '=>', entity
                self.lexicon[synonym].append((entity, type))

    def add_numbers(self):
        numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        for i, n in enumerate(numbers):
            for phrase in [str(i), n]:
                self.lexicon[phrase].append((str(i), 'number'))

    def test(self):
        phrases = ['i', 'physics', 'comp sci', 'econ', 'penn', 'cs', 'upenn', 'u penn', 'u of p', 'ucb', 'berekely', 'jessica']
        phrases = ['foodie', 'evening', 'evenings', 'food']
        for x in phrases:
            print x, '=>', self.lookup(x)
