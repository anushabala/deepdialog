__author__ = 'anushabala'

__author__ = 'anushabala'

import numpy as np
from argparse import ArgumentParser
import json
from collections import defaultdict
import os
import uuid


class Friend(object):
    def __init__(self, name, indoors, morning, hobby):
        self.name = name
        self.indoors = {'name': indoors}
        self.morning = {'name': morning}
        self.hobby = {'name': hobby}

    def __info__(self):
        return {
            "name": self.name,
            "indoors": self.indoors,
            "morning": self.morning,
            "hobby": self.hobby,
        }

    def same_morning(self, other_friend):
        return self.morning["name"] == other_friend.morning["name"]

    def same_indoors(self, other_friend):
        return self.indoors["name"] == other_friend.indoors["name"]

    def same_hobbies(self, other_friend):
        return self.hobby["name"] == other_friend.hobby["name"]

    def compatible(self, other_friend):
        return self.same_indoors(other_friend) and self.same_hobbies(other_friend) and self.same_morning(other_friend)


class FriendNetwork(object):
    friend_p = 0.1

    def __init__(self):
        self.friends = []
        self.relationships = defaultdict(list)

    def __info__(self):
        return [friend.__info__() for friend in self.friends]

    def add_relationship(self, user1, user2):
        self.relationships[user1].append(user2)
        self.relationships[user2].append(user1)

    def add_friend(self, friend):
        self.friends.append(friend)

    def find_potential_friends(self):
        potential_friends = []
        for other_friend in self.friends:
            if np.random.rand() <= self.friend_p:
                potential_friends.append(other_friend)
        return potential_friends

    def find_compatible_friends(self, friend, other_friend):
        connections = []
        first_friends = self.relationships[friend]
        second_friends = self.relationships[other_friend]
        for a in first_friends:
            for b in second_friends:
                if a not in second_friends and b not in first_friends:
                    if a.compatible(b):
                        connections.append((friend, a, other_friend, b))

        return connections

    def find_scenario_candidates(self):
        candidates = []
        added_friends = []
        for person in self.friends:
            added_friends.append(person)
            for other_person in self.friends:
                if other_person not in added_friends:
                    connections = self.find_compatible_friends(person, other_person)
                    candidates.extend(connections)

        return candidates


class NetworkGenerator(object):
    indoors_choices = ['Indoors','Outdoors']
    morning_choices = ['Morning', 'Evening']
    hobbies = ['Hiking', 'Road Trips', 'Snorkeling', 'Foodie', 'Movies', 'Traveling', 'Reading', 'Cooking']

    def generate_random_friend(self, name):
        indoors = np.random.choice(self.indoors_choices)
        morning = np.random.choice(self.morning_choices)
        hobby = np.random.choice(self.hobbies)
        return Friend(name, indoors, morning, hobby)

    def __init__(self, N=50, names_file='data/person_names.txt'):
        self.network_size = N
        self.names = list(set([line.strip() for line in open(names_file, 'r').readlines()]))
        self.network = FriendNetwork()

    def create_network(self):
        selected_names = np.random.choice(self.names, self.network_size, replace=False)
        for name in selected_names:
            friend = self.generate_random_friend(name)
            self.network.add_friend(friend)
            potential_friends = self.network.find_potential_friends()
            self.create_relationships(friend, potential_friends)

    def create_relationships(self, friend, potential_friends):
        for other_friend in potential_friends:
            print "adding relationship between %s and %s" % (friend.name, other_friend.name)
            self.network.add_relationship(friend, other_friend)


class ScenarioGenerator(object):
    def __init__(self, network):
        self.network = network
        self.scenario_candidates = network.find_scenario_candidates()

    def generate_scenario(self, num_friends=50):
        scenario = {}
        candidates = self.scenario_candidates[np.random.choice(xrange(0, len(self.scenario_candidates)))]
        user1 = candidates[0]
        first_friend = candidates[1]
        user2 = candidates[2]
        second_friend = candidates[3]
        user1_friends = [first_friend]
        user2_friends = [second_friend]
        scenario["agents"] = [{"info": user1.__info__(),
                               "connection": first_friend.__info__()},
                              {"info": user2.__info__(),
                               "connection": second_friend.__info__()}]

        # add all friends of each user except their mutual friends (apart from the one common connection)
        user1_friends.extend([f for f in self.network.relationships[user1] if f not in self.network.relationships[user2] and f not in user1_friends and f !=second_friend and f!=first_friend and not f.compatible(second_friend)])
        for other_friend in self.network.relationships[user2]:
            if other_friend in user1_friends:
                continue
            not_compatible_at_all = True
            for possible_match in user1_friends:
                if other_friend.compatible(possible_match):
                    not_compatible_at_all = False
                    break
            if not_compatible_at_all:
                user2_friends.append(other_friend)

        if len(user1_friends) > num_friends:
            user1_friends = user1_friends[:num_friends]

        if len(user2_friends) > num_friends:
            user2_friends = user2_friends[:num_friends]

        ctr = 0
        print len(user1_friends), len(user2_friends)
        while ctr < len(self.network.friends) and len(user1_friends) < num_friends:
            friend = self.network.friends[ctr]
            if friend == user1 or friend == user2 or friend == first_friend or friend == second_friend:
                ctr += 1
                continue
            if friend in user1_friends or friend in user2_friends:
                ctr +=1
                continue
            not_compatible_at_all = True
            for possible_match in user2_friends:
                if friend.compatible(possible_match):
                    not_compatible_at_all = False

            if not_compatible_at_all:
                user1_friends.append(friend)
                ctr += 1
            else:
                ctr += 1

        ctr = 0
        while ctr < len(self.network.friends) and len(user2_friends) < num_friends:
            friend = self.network.friends[ctr]
            if friend == user1 or friend == user2 or friend == first_friend or friend == second_friend:
                ctr += 1
                continue
            if friend in user1_friends or friend in user2_friends:
                ctr +=1
                continue
            not_compatible_at_all = True
            for possible_match in user1_friends:
                if friend.compatible(possible_match):
                    not_compatible_at_all = False

            if not_compatible_at_all:
                user2_friends.append(friend)
                ctr += 1
            else:
                ctr+=1

        # print len(user1_friends), len(user2_friends)
        np.random.shuffle(user1_friends)
        np.random.shuffle(user2_friends)
        scenario["agents"][0]["friends"] = [f.__info__() for f in user1_friends]
        scenario["agents"][1]["friends"] = [f.__info__() for f in user2_friends]


        # make some assertion here
        candidates = 0
        for a in user1_friends:
            for b in user2_friends:
                if a.compatible(b):
                    candidates += 1

        common = [f for f in user1_friends if f in user2_friends]

        try:
            assert candidates == 1
            assert len(common) == 0
        except AssertionError:
            print "First friend"
            print first_friend.__info__()
            print "Second friend"
            print second_friend.__info__()
            print "User 1 friends"
            for a in user1_friends:
                print a.__info__()
            print "User 2 friends"
            for b in user2_friends:
                print b.__info__()

        return scenario


def write_user(info, outfile, fewer_lines=False):
    outfile.write("\tName: %s" % info["name"])
    outfile.write("\n")
    outfile.write("\tIndoors/Outdoors: %s" % info["indoors"]["name"])
    outfile.write("\tMorning/Evening: %s" % info["morning"]["name"])
    outfile.write("\n")
    outfile.write("\tHobby: %s" % ", ".join(info["hobby"]["name"]))
    outfile.write("\n\n")


def write_scenario_to_readable_file(scenario, user1_file, user2_file):
    write_user(scenario["agents"][0]["info"], user1_file)
    user1_file.write("Friends:\n")
    for f in scenario["agents"][0]["friends"]:
        write_user(f, user1_file, fewer_lines=True)
        user1_file.write("\n")
    write_user(scenario["agents"][1]["info"], user2_file)
    user2_file.write("Friends:\n")
    for f in scenario["agents"][1]["friends"]:
        write_user(f, user2_file, fewer_lines=True)
        user2_file.write("\n")


def write_scenarios_to_json(scenarios, json_file):
    json.dump(scenarios, open(json_file, 'w'))


def write_json_to_file(network, outfile):
    json_network = json.dumps(network.__info__())
    outfile.write(json_network+"\n")


def main(args):
    outfile = open(args.output, 'w')

    num_scenarios = args.num_scenarios
    generator = NetworkGenerator(args.size)
    generator.create_network()
    scenarios = []
    for i in xrange(0, num_scenarios):
        scen_file_1 = open(os.path.join(args.scenario_dir, 'scenario%d_User1.out' % i,), 'w')
        scen_file_2 = open(os.path.join(args.scenario_dir, 'scenario%d_User2.out' % i,), 'w')
        scenario_gen = ScenarioGenerator(generator.network)
        scenario = scenario_gen.generate_scenario(num_friends=6)
        scenario["uuid"] = str(uuid.uuid4())
        scenarios.append(scenario)
        write_scenario_to_readable_file(scenario, scen_file_1, scen_file_2)
        # write_json_to_file(generator.network, args.output)
        scen_file_1.close()
        scen_file_2.close()

    write_scenarios_to_json(scenarios, args.output)
    outfile.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, default=100, help='Size of network to generate')
    parser.add_argument('--output', type=str, default='data/scenarios.json', help='File to write networks to.')
    parser.add_argument('--num_scenarios', type=int, default=100, help='Number of scenarios to generate')
    parser.add_argument('--scenario_dir', default='data/scenarios', help='File to write scenario to')

    clargs = parser.parse_args()
    main(clargs)