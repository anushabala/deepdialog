class KB(object):
    '''
    Main job: read scenarios and turn them into tables.
    '''
    def __init__(self, scenario, agent):
        # Convert JSON format into a nicer table format so we can do joins
        self.scenario_id = scenario['uuid']
        self.agent = agent

        self.table = []  # List of people (dictionary mapping attribute to person
        def normalize(s): return s.encode('utf-8').lower()
        def convert(person_info):
            row = {}
            row['Name'] = (normalize(person_info['name']), 'person')
            # MutualFriends
            if 'company' in person_info:
                row['Company'] = (normalize(person_info['company']['name']), 'company')
            if 'school' in person_info:
                row['School'] = (normalize(person_info['school']['name']), 'school')
                row['Major'] = (normalize(person_info['school']['major']), 'major')
            # Matchmaking
            if 'morning' in person_info:
                row['TimePref'] = (normalize(person_info['morning']['name']), 'timePref')
            if 'indoors' in person_info:
                row['LocPref'] = (normalize(person_info['indoors']['name']), 'locPref')
            if 'hobby' in person_info:
                row['Hobby'] = (normalize(person_info['hobby']['name']), 'hobby')
            return row

        agent_info = scenario['agents'][agent]  # Information specific to agent
        self.table.append(convert(agent_info['info']))
        for friend in agent_info['friends']:
            self.table.append(convert(friend))

        # Compute types and relations
        self.types = sorted(list(set(v[1] for v in self.table[0].values())))  # e.g., person
        self.relations = sorted(list(k for k in self.table[0].keys() if k != 'Name'))  # School

    def dump(self):
        print '############ KB %s, agent=%s:' % (self.scenario_id, self.agent,)
        all_relations = ['Name'] + self.relations
        widths = [max(len(str(row[rel])) for row in self.table) for rel in all_relations]
        for row in [dict((rel, rel) for rel in all_relations)] + self.table:
            print ' ', ' '.join(('%%-%ds' % widths[i]) % (row[rel],) for i, rel in enumerate(all_relations))
