__author__ = 'anushabala'
import re
import json
from chat.modeling import tokens as mytokens

user_regex = r'User ([0-9])'
restaurant_regex = 'Selected *[0-9]*: (.*)'
user_id_regex = r'User ([0-9]) has user ID (.*)'
SELECT_RESTAURANT = 'Selected restaurant:'
NO_OUTCOME = "NO_OUTCOME"
COMPLETE = 'COMPLETE'
TOO_SHORT = "TOO_SHORT"
SHORT = "SHORT"
BOTS = ["DEFAULT_BOT", "LSTM_UNFEATURIZED", "LSTM_FEATURIZED"]
HUMAN = "human"
BOT_METADATA = ["TAGS", "SYNONYMS", "PROBABILITIES"]


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


def parse_transcript(scenarios, transcript_file, include_bots=False):
    infile = open(transcript_file, 'r')
    transcript = {}
    choices = {}
    ids = {}
    dialogue = []
    transcript["ids"] = ids
    transcript["choices"] = choices
    transcript["dialogue"] = dialogue
    BOT_NUM = None

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
            agent_num = line[2]

            if line[2] in BOTS:
                BOT_NUM = agent_num
                transcript["BOT_TYPE"] = line[2]
                if not include_bots:
                    return None
            elif line[2].strip() in BOT_METADATA:
                if not include_bots:
                    return None
                else:
                    continue
            else:
                transcript["BOT_TYPE"] = HUMAN
                user_match = re.match(user_regex, line[2])
                agent_num = int(user_match.group(1))
                # print agent_num
            if len(line) == 5:
                choice_match = re.match(restaurant_regex, " ".join(line[3:]))
                if choice_match:
                    choices[agent_num] = choice_match.group(1)
                    # print line
                    # print agent_num
                    # print choices[agent_num]
                    dialogue.append((agent_num, mytokens.SELECT_NAME + ' ' + choices[agent_num].lower()))
                elif line[3] == SELECT_RESTAURANT:
                    choices[agent_num] = line[4]
            else:
                try:
                    dialogue.append((agent_num, line[3]))
                except IndexError:
                    continue

    transcript["outcome"] = NO_OUTCOME
    if include_bots and BOT_NUM is not None:
        keys = choices.keys()

        # print keys
        for key in keys:
            if key != BOT_NUM and BOT_NUM in choices.keys() and choices[key] == choices[BOT_NUM]:
                transcript["outcome"] = choices[key]
    else:
        scenario = scenarios[transcript["scenario"]]
        agents = scenario["agents"]
        if 0 in choices.keys() and 1 in choices.keys():
            if 'connection' in agents[0]:  # For Matchmaking
                if choices[0] == agents[0]["connection"]["name"] and choices[1] == agents[1]["connection"]["name"]:
                    transcript["outcome"] = COMPLETE
            else:  # For MutualFriends
                if choices[0] == choices[1]:
                    transcript["outcome"] = COMPLETE

    infile.close()
    return transcript
