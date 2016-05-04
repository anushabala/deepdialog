__author__ = 'anushabala'
import re
import json

user_regex = r'User ([0-9])'
restaurant_regex = 'Selected [0-9]+: (.*)'
user_id_regex = r'User ([0-9]) has user ID (.*)'
SELECT_RESTAURANT = 'Selected restaurant:'
NO_OUTCOME = "NO_OUTCOME"
TOO_SHORT = "TOO_SHORT"
SHORT = "SHORT"
BOT = "BOT"
BOT_METADATA = ["TAGS", "SYNONYMS", "PROBABILITIES"]
BOT_NUM = -1


def load_scenarios(scenarios_file):
    json_scenarios = json.load(open(scenarios_file, 'r'), encoding='utf-8')
    return {scenario["uuid"]:scenario for scenario in json_scenarios}


def is_transcript_valid(transcript):
    if transcript["outcome"] == NO_OUTCOME:
        return False, NO_OUTCOME
    return True, None


def is_transcript_short(transcript):
    user_0_count = len([d for d in transcript["dialogue"] if d[0] == 0])
    user_1_count = len([d for d in transcript["dialogue"] if d[0] == 1])

    if user_0_count < 1 or user_1_count < 1:
        return True, TOO_SHORT

    if user_0_count < 2 or user_1_count < 2:
        return True, SHORT

    return False, None


def parse_transcript(transcript_file, include_bots=False):
    infile = open(transcript_file, 'r')
    transcript = {}
    choices = {}
    ids = {}
    dialogue = []
    transcript["ids"] = ids
    transcript["choices"] = choices
    transcript["dialogue"] = dialogue

    for line in infile.readlines():
        line = line.strip().split("\t")
        if line[1] == '---' or (len(line) == 4 and line[-1] == 'joined'):
            continue
        if "scenario" not in transcript.keys():
            scenario_id = line[1]
            transcript["scenario"] = scenario_id

        id_match = re.match(user_id_regex, line[2])
        if id_match:
            agent_num = int(id_match.group(1))
            ids[agent_num] = id_match.group(2)
        else:
            agent_num = BOT_NUM
            if line[2] == BOT:
                if not include_bots:
                    return None
            elif line[2].strip() in BOT_METADATA:
                if not include_bots:
                    return None
                else:
                    continue
            else:
                user_match = re.match(user_regex, line[2])
                agent_num = int(user_match.group(1))
            if len(line) == 5:
                choice_match = re.match(restaurant_regex, " ".join(line[3:]))
                if choice_match:
                    choices[agent_num] = choice_match.group(1)
                    dialogue.append((agent_num, "SELECT NAME "+choices[agent_num]))
                elif line[3] == SELECT_RESTAURANT:
                    choices[agent_num] = line[4]
            else:
                try:
                    dialogue.append((agent_num, line[3]))
                except IndexError:
                    continue

    transcript["outcome"] = NO_OUTCOME
    if include_bots:
        keys = choices.keys()
        for key in keys:
            if key != BOT_NUM and choices[key] == choices[BOT_NUM]:
                transcript["outcome"] = choices[key]
    else:
        if 0 in choices.keys() and 1 in choices.keys() and choices[0] == choices[1]:
            transcript["outcome"] = choices[0]

    infile.close()
    return transcript
