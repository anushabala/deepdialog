import math
from chat.lib import sample_utils
from chat.modeling import tokens as mytokens

def sample_trajectory(states):
    '''
    Works with the dataset.
    A state encodes a weighted set of candidate formula_tokens.  Sample one of them.
    Return the sampled [(who, formula_tokens)] along with its log(weight).
    '''
    log_weight = 0
    state = sample_utils.sample_candidates([(state, state['weight']) for state in states])[0]
    log_weight += math.log(state['weight'])
    messages = []
    for message in state['messages']:
        formula_tokens = []
        for candidates in message['formula_token_candidates']:
            if not isinstance(candidates, list):
                formula_tokens.append(candidates)
            else:
                token, weight = sample_utils.sample_candidates(candidates)
                formula_tokens.append(token)
                log_weight += math.log(weight)
        messages.append((message['who'], formula_tokens))
    return (messages, log_weight)

def messages_to_sequences(agent, messages):
    '''
    Encode the list of (who, tokens) pairs into pairs of sequences for training.
    Input: [(0, hi), (0, i know 2 people at facebook), (1, cool), (0, SELECT_NAME alice), (1, SELECT_NAME alice)]
    Output: [[END_TURN], [SAY, hi, END, SAY, i, ...], ...]
        Partner                     Agent
        =======                     =====
        END_TURN                    SAY hi END
                                    SAY i know 2 people at facebook END_TURN
        SAY cool END_TURN           SELECT_NAME alice END_TURN
        SELECT_NAME alice END_TURN  END_TURN
    Note that we pad with END_TURN to make the sequences returned:
        [partner, agent, ..., partner, agent]
    '''
    seqs = []
    curr_seq = [None]
    def add(token):
        if curr_seq[0] is None:
            curr_seq[0] = []
            seqs.append(curr_seq[0])
        curr_seq[0].append(token)
        if token == mytokens.END_TURN:
            curr_seq[0] = None

    if messages[0][0] != 1 - agent:  # Pad so partner starts
        add(mytokens.END_TURN)
    for i, (who, tokens) in enumerate(messages):
        if tokens[0] != mytokens.SELECT_NAME:
            add(mytokens.SAY)
        for token in tokens:
            add(token)
        if i+1 == len(messages) or messages[i+1][0] != who:
            add(mytokens.END_TURN)
        else:
            add(mytokens.END)
    if messages[-1][0] != agent:  # Pad so agent ends
        add(mytokens.END_TURN)

    return seqs
