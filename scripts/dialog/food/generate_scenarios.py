import sys
import random
import json
import uuid
import numpy as np
import argparse

class Restaurant(object):
    def __init__(self, line, price_ranges, price_range_probs=None):
        if price_range_probs is not None:
            price_range_probs = np.ones(len(price_ranges))/float(len(price_ranges))

        tokens = line.strip().split("\t")
        if len(tokens)==2:
            self.cuisine, self.name = tokens
            i = np.random.choice(len(price_ranges), p=price_range_probs)
            self.price_range = price_ranges[i]
        elif len(tokens)==3:
            self.cuisine, price_range, self.name = tokens
            self.price_range = tuple(map(lambda x:int(x), price_range.split("-")))
        else:
            raise Exception("Unexpected format for line: {}".format(line))

    def __info__(self):
        return (self.name, self.cuisine, self.price_range)

        

class World(object):
    def __init__(self, config):
        self.restaurants = []

        self.price_ranges = [tuple(i) for i in config["price_ranges"]]
        self.price_range_probs = config["price_range_probabilities"]
        with open(config["restaurants_file"]) as fin:
            for line in fin:
                self.restaurants.append(Restaurant(line, self.price_ranges, self.price_range_probs))
        self.restaurants = sorted(self.restaurants, key=lambda x:x.name)
        self.cuisines = sorted(set(r.cuisine for r in self.restaurants))

    def __info__(self):
        d = {}
        d["price_ranges"] = self.price_ranges
        d["restaurants"] = [r.__info__() for r in self.restaurants]
        d["cuisines"] = self.cuisines
        return d

class SpendingFunc(object):
    def __init__(self, u, m):
        self.utility = u
        self._map = m

    def f(self, r):
        return self.utility[self._map[r]]

    def get_preference_list(self):
        results = []
        for k in sorted(self._map.keys(), key=lambda x: -self.utility[self._map[x]]):
            results.append((k,self.utility[self._map[k]]))
        return results

    def __info__(self):
        results = []
        for k in sorted(self._map.keys(), key=lambda x: -self.utility[self._map[x]]):
            results.append((k,self.utility[self._map[k]]))
        return results


class SpendingFuncFactory(object):
    def __init__(self, ranges, u_match, u_exp, u_cheap, u_dtype=np.int_):
        self.ranges = ranges
        self.u_match = u_match
        self.u_exp = u_exp
        self.u_cheap = u_cheap
        self.u_dtype = u_dtype

    def create_from_optimal_index(self, index):
        assert index < len(self.ranges)
        u = np.zeros(len(self.ranges), dtype=self.u_dtype)
        u[:index] = self.u_cheap
        u[index] = self.u_match
        u[index+1:] = self.u_exp
        m = {item:index for index,item in enumerate(self.ranges)}
        return SpendingFunc(u, m)

class CuisineFunc(object):
    def __init__(self, _f, m):
        self._f = _f
        self.m = m

    def f(self, c):
        return self._f[self.m[c]]

    def get_preference_list(self):
        results = []
        for k in sorted(self.m.keys(), key=lambda x: -self._f[self.m[x]]):
            results.append((k,self._f[self.m[k]]))
        return results        

    def __info__(self):
        results = []
        for k in sorted(self.m.keys(), key=lambda x: -self._f[self.m[x]]):
            results.append((k,self._f[self.m[k]]))
        return results

class CuisineFuncFactory(object):
    def __init__(self, index_to_utility):
        self.index_to_utility = index_to_utility

    def create_from_ordering(self, ordering):
        _map = {item:index for index,item in enumerate(ordering)}
        return CuisineFunc(self.index_to_utility,_map)
        

class Agent(object):
    def __init__(self, cuisine_func, spending_func):
        self.cuisine_func = cuisine_func
        self.spending_func = spending_func

    def utility(self, restaurant):
        return self.cuisine_func.f(restaurant.cuisine) + self.spending_func(restaurant.price_range)

    def __info__(self):
        i = {}
        i["spending_func"] = self.spending_func.__info__()
        i["cuisine_func"] = self.cuisine_func.__info__()
        return i

class Scenario(object):
    def __init__(self, cuisines, restaurants, agents):
        self.cuisines = cuisines
        self.restaurants = restaurants
        self.agents = agents
        self.uuid = str(uuid.uuid4())

    def __info__(self):
        d = {}
        d["cuisines"] = self.cuisines
        d["restaurants"] = [r.__info__() for r in self.restaurants]
        d["agents"] = [a.__info__() for a in self.agents]
        d["uuid"] = self.uuid
        return d
            

class ScenarioMaker(object):
    def __init__(self, world, config):
        self.world = world
        self.num_cuisines = config["num_cuisines"]
        self.num_restaurants = config["num_restaurants"]
        self.num_agents = config["num_agents"]
        self.randgen = random.Random(config["random_seed"])
        c = config["cuisine_func_factory"]
        self.cfactory = CuisineFuncFactory(c["index_to_utility"])
        c = config["spending_func_factory"]
        self.sfactory = SpendingFuncFactory(world.price_ranges, c["utility_if_match"], c["utility_if_more_expensive"], c["utility_if_cheaper"])

    def make(self):
        cuisines = self.randgen.sample(self.world.cuisines, self.num_cuisines)
        
        matching_restaurants = [r for r in self.world.restaurants if r.cuisine in cuisines]

        restaurants = self.randgen.sample(matching_restaurants, min(self.num_restaurants,len(matching_restaurants)))
        
        agents = []
        for i in range(self.num_agents):
            #cuisine_ordering = self.world.cuisines[:]
            cuisine_ordering = cuisines[:]
            self.randgen.shuffle(cuisine_ordering)

            cf = self.cfactory.create_from_ordering(cuisine_ordering)
            sf = self.sfactory.create_from_optimal_index(self.randgen.randrange(len(self.world.price_ranges)))
            agents.append(Agent(cf,sf))

        return Scenario(cuisines, restaurants, agents)

def main():
    parser = argparse.ArgumentParser()
    
    # Input location and mode
    parser.add_argument("-config", type=str, required=True, help="input json config file")
    parser.add_argument("-output_prefix", type=str, required=True, help="Output file prefix")
    args = parser.parse_args()

    config_file = args.config

    with open(config_file) as fn:
        config = json.load(fn)

    world = World(config["world"])
    scenario_maker = ScenarioMaker(world, config["scenario_maker"])
    n = config["num_scenarios"]
    TAB="    "
    scenario_objs = []
    with open("{}.info.json".format(args.output_prefix),"w") as fout:
        with open("{}.agent.1.txt".format(args.output_prefix),"w") as f1:
            with open("{}.agent.2.txt".format(args.output_prefix),"w") as f2:
                for i in range(n):
                    s = scenario_maker.make()
                    scenario_objs.append(s.__info__())

                    for f in [f1,f2]:
                        f.write("====================================================================================\n")
                        f.write("======= Scenario #{}  ==============================================================\n".format(i))
                        f.write("====================================================================================\n")
                        f.write("\n")
                        f.write("Restaurants:\n")
                        for r in sorted(s.restaurants, key=lambda x:(x.cuisine,x.name)):
                            f.write("{}{:25}\t{:25}\t${}-${}\n".format(TAB, r.name, r.cuisine, r.price_range[0],r.price_range[1]))

                        for a,f in [(s.agents[0],f1),(s.agents[1],f2)]:
                            f.write("\n")
                            f.write("Agent Profile:\n")
                            f.write("{}Preferred Cuisine Order:\n".format(TAB))
                            for c,score in a.cuisine_func.get_preference_list():
                                if c in s.cuisines:
                                    f.write("{}{}{}\n".format(TAB,TAB,c))
                            f.write("{}Preferred Price Range:\n".format(TAB))
                            for p,score in a.spending_func.get_preference_list():
                                f.write("{}{}${}-${}\n".format(TAB,TAB,p[0],p[1]))
                            f.write("\n")
        json.dump(scenario_objs, fout)
                  


if __name__ == "__main__":
    main()

