__author__ = 'anushabala'
import numpy as np

START_TOKEN = "_start_dialogue_"
SELECT_TOKEN = "_select_name_"
SAY_TOKEN = "_say_"
PASS_TOKEN = "_action_pass_" 


def sample_state(example):
    probs = [state["prob"] for state in example["states"]]
    choice = np.random.choice(xrange(len(example["states"])), p=probs)

    return example["states"][choice]


def sample_logical_form(message):
    sampled_tokens = []
    for candidate_set in message['formula_token_candidates']:
        if len(candidate_set) == 1:
            sampled_tokens.append(candidate_set[0][0])
        else:
            candidate_tokens = [candidate[0] for candidate in candidate_set]
            probs = [candidate[1] for candidate in candidate_set]
            token = np.random.choice(candidate_tokens, p=probs)
            sampled_tokens.append(token)
    return sampled_tokens


def create_sequences_from_dialogue(dialogue, this_agent):
    #print dialogue
    #print type(this_agent)
    first_agent = dialogue[0][0]
    #print type(first_agent)
    other_agent = 1 - this_agent
    #print first_agent
    #print this_agent
    #print other_agent
    #print first_agent == this_agent
    if first_agent != this_agent:
        dialogue[0][1].insert(0, START_TOKEN)
        current_agent = first_agent

    else:
        dialogue.insert(0, (other_agent, [START_TOKEN]))
        current_agent = other_agent

    #print dialogue
    x = []
    y = []
    current_seqs = []
    for (agent, tokens) in dialogue:
        if tokens[0] != SELECT_TOKEN and tokens[0] != START_TOKEN:
            tokens.insert(0, SAY_TOKEN)

        seq = " ".join(tokens)
        if current_agent != agent:
            agent_list = y if current_agent == this_agent else x
            agent_list.append(" ".join(current_seqs))
            current_agent = agent
            current_seqs = []
        current_seqs.append(seq)

    agent_list = y if current_agent == this_agent else x
    agent_list.append(" ".join(current_seqs))

    if current_agent == other_agent and len(x) == len(y)+1:
        y.append(PASS_TOKEN)
    return x, y


def sample_dialogue(example):
    selected_state = sample_state(example)
    this_agent = example["agent"]
    dialogue = []
    for message in selected_state["messages"]:
        agent_idx = message["who"]
        tokens = sample_logical_form(message)
        dialogue.append((agent_idx, tokens))

    return create_sequences_from_dialogue(dialogue, this_agent)
