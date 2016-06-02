import codecs
from collections import defaultdict
import string
import operator

__author__ = 'anushabala'
from argparse import ArgumentParser
import json
import os
from chat.modeling.tagger import Entity
from chat.modeling.chatbot import ChatBot
from chat.modeling.lexicon import load_scenarios
import numpy as np

SEQ_DELIM = "</s>"
SAY_DELIM = "SAY "
START = "START"
SELECT = "SELECT NAME"
COL_WIDTH="\"30%%\""
OUTPUT_STYLE="\"color:#4400FF\""
REGULAR_STYLE="\"color:#000000\""
MENTION_WINDOW = 3


def split_seq_into_sentences(seqs):
    seqs = seqs.replace("<", "&lt;")
    seqs = seqs.replace(">", "&gt;")
    seqs = seqs.split(SAY_DELIM)
    print "splitting", seqs
    if seqs[0].strip() == START and len(seqs) > 1:
        seqs.pop(0)
    new_seqs = []
    for s in seqs:
        if SELECT in s:
            i = s.index(SELECT)
            new_seqs.append(s[:i])
            new_seqs.append(s[i:])
        else:
            new_seqs.append(s)
    return new_seqs


def replace_my_entities(s, bot):
    pass


def get_tag_and_features(token):
    if "<" not in token and ">" not in token:
        # not a tagged entity, return nonetypes
        return None, None

    token = token.split(">_<")
    assert len(token) > 0

    tag = token[0].strip("<>")
    if len(token) == 1:
        return tag, None

    features = [t.strip("<>") for t in token[1:]]
    return tag, features


def get_all_friend_info_with_tag(tag, json_info):
    if tag == Entity.to_tag(Entity.FULL_NAME):
        return list({friend["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.COMPANY_NAME):
        return list({friend["company"]["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.SCHOOL_NAME):
        return list({friend["school"]["name"].lower() for friend in json_info["friends"]})
    if tag == Entity.to_tag(Entity.MAJOR):
        return list({friend["school"]["major"].lower() for friend in json_info["friends"]})


def rank_friends(mentioned_by_partner, bot_info, choices):
    friends = bot_info["friends"]
    print "Ranking friends. Choices: ", choices
    print "Mentions: ", mentioned_by_partner
    probs = {f:0 for f in choices}
    for f in friends:
        name = f["name"].lower()
        if name not in choices:
            continue
        if f["school"]["name"].lower() in mentioned_by_partner[Entity.to_tag(Entity.SCHOOL_NAME)]:
            probs[name] += 1
        if f["school"]["major"].lower() in mentioned_by_partner[Entity.to_tag(Entity.MAJOR)]:
            probs[name] += 1
        if f["company"]["name"].lower() in mentioned_by_partner[Entity.to_tag(Entity.COMPANY_NAME)]:
            probs[name] += 1
    print "Scores:",probs

    ranked_choices = [a[0] for a in sorted(probs.items(), key=operator.itemgetter(1), reverse=True)]
    return ranked_choices


def replace_entities_in_seq(sequence, scenario, bot_idx, current_user, mentioned_by_bot, mentioned_by_partner):
    print "My mentions: ", mentioned_by_bot
    print "Friend mentions:", mentioned_by_partner
    print ""
    agent_info = scenario["agents"][current_user]
    bot_info = scenario["agents"][bot_idx]
    other_user_info = scenario["agents"][1-current_user]
    print "Current agent (%d) info:" % current_user, agent_info
    print "Bot (%d) info" % bot_idx, bot_info
    # sequence = str(sequence).translate(string.maketrans("",""), string.punctuation)
    tokens = sequence.strip().split()
    new_sentence = []
    new_mentions = defaultdict(list)
    for token in tokens:
        tag, features = get_tag_and_features(token)
        if tag is None:
            print "No tag or features: ", token
            new_sentence.append(token)
            continue
        print token, tag, features
        all_mentions = []
        all_mentions.extend(mentioned_by_bot[tag])
        all_mentions.extend(mentioned_by_partner[tag])
        all_mentions.extend(new_mentions[tag])
        entity = None
        if features is None:
            # no match or mention feature
            friend_entities = get_all_friend_info_with_tag(tag, agent_info)
            choices = [e for e in friend_entities if e not in all_mentions]
            if len(choices) == 0:
                print "Invalid tag generated (all entities mentioned and no _MENTIONED_ features generated), " \
                      "choosing random entity"
                entity = np.random.choice(friend_entities)
            else:
                entity = np.random.choice(choices)
        else:
            # features depend on tag type
            choices = []
            # mentions features possible for all entity types
            if "F:MENTIONED_BY_FRIEND" in features and "F:MENTIONED_BY_ME" in features:
                choices = [f for f in mentioned_by_bot[tag] if f in mentioned_by_partner[tag]]
                if len(choices) == 0:
                    # todo what to do here?
                    choices = all_mentions
                print "Found both mention features! All mentions", choices
            elif "F:MENTIONED_BY_FRIEND" in features and "F:MENTIONED_BY_ME" not in features:
                choices = [f for f in mentioned_by_partner[tag] if f not in mentioned_by_bot[tag]]
                if len(choices) == 0:
                    all_entities = get_all_friend_info_with_tag(tag, agent_info)
                    choices = all_entities
                    if current_user != bot_idx:
                        # mentions are empty, tagging not consistent with model out
                        pass
                    else:
                        # print warning? this shouldn't really happen.. todo see if these cases are legit
                        print "WARNING: generated mention tag when no mentions were made"
                print "Found friend mention:", tag, mentioned_by_partner[tag], mentioned_by_bot[tag], choices
            elif "F:MENTIONED_BY_ME" in features and "F:MENTIONED_BY_FRIEND" not in features:
                choices = [f for f in mentioned_by_bot[tag] if f not in mentioned_by_partner[tag]]
                if len(choices) == 0:
                    all_entities = get_all_friend_info_with_tag(tag, agent_info)
                    choices = all_entities
                    if current_user != bot_idx:
                        # mentions are empty, tagging not consistent with model output
                        pass
                    else:
                        print "WARNING: generated mention tag when no mentions were made"
                print "Found my mention:", tag, mentioned_by_partner[tag], mentioned_by_bot[tag], choices
            else:
                # no mentions
                all_entities = get_all_friend_info_with_tag(tag, agent_info)
                if current_user != bot_idx:
                    choices = all_entities
                else:
                    # bot should always generate a mentioned tag unless not referencing? todo figure out whether to enforce grammar
                    choices = all_entities
                print "No mentions"
                print "Tag: %s" % tag
                print "All entities:", all_entities
                print "Choices:", choices

            # check for other features
            if tag == Entity.to_tag(Entity.FIRST_NAME):
                # only known/unknown features possible
                if current_user == bot_idx:
                    # if current user is bot, use bot info
                    if "F:UNKNOWN" in features:
                        # pass - do nothing? there has to be a user mention here
                        # maybe do an assertion
                        print "WARNING: generated friend with F:UNKNOWN, choices (based on mentions) are", choices
                        print "Mentions: ", all_mentions
                        entity = ""
                    elif "F:KNOWN" in features:
                        print "KNOWN FRIEND, BOT UTTERNACE"
                        my_friends = get_all_friend_info_with_tag(tag, agent_info)
                        partner_friends = get_all_friend_info_with_tag(tag, other_user_info)
                        print "Partner friends", partner_friends
                        print "Bot friends: ", my_friends
                        choices = [f for f in choices if f in my_friends]
                        if len(choices)==0:
                            # accidentally generrated a mention tag? todo hacky
                            choices = [f for f in my_friends if f not in all_mentions]

                        choices = rank_friends(mentioned_by_partner, bot_info, choices)
                        print "Choices:", choices
                        entity = choices[0]
                        if "F:MENTIONED_BY_FRIEND" in features and scenario["connection"]["info"]["name"].lower() in mentioned_by_partner[tag]:
                            entity = scenario["connection"]["info"]["name"].lower()
                else:
                    if "F:UNKNOWN" in features:
                        print "UNKNOWN FRIEND, AGENT UTTERANCE"
                        my_friends = get_all_friend_info_with_tag(tag, agent_info)
                        bot_friends = get_all_friend_info_with_tag(tag, bot_info)
                        print "Agent friends", my_friends
                        print "Bot friends", bot_friends
                        choices = [f for f in choices if f in my_friends if f not in bot_friends]
                        if "F:MENTIONED_BY_ME" in features and len(choices) == 0:
                            # tag in agent utterance doesn't sync with mentions
                            choices = my_friends
                        print "Choices", choices
                        entity = np.random.choice(choices)
                    else:
                        # connection!!!
                        entity = scenario["connection"]["info"]["name"]
            else:
                #only matches possible
                if "F:MATCH_ME" in features:
                    if tag == Entity.to_tag(Entity.MAJOR):
                        entity = bot_info["info"]["school"]["major"].lower()
                    elif tag == Entity.to_tag(Entity.SCHOOL_NAME):
                        entity = bot_info["info"]["school"]["name"].lower()
                    elif tag == Entity.to_tag(Entity.COMPANY_NAME):
                        entity = bot_info["info"]["company"]["name"].lower()
                elif "F:MATCH_FRIEND" in features:
                    print "Feature matches bot friend: ", tag, features, token
                    all_entities = get_all_friend_info_with_tag(tag, bot_info)
                    print "Bot entities for tag:", all_entities
                    print "Existing choices", choices
                    choices = [e for e in choices if e in all_entities]
                    if len(choices) == 0 and ("F:MENTIONED_BY_FRIEND" in features or "F:MENTIONED_BY_ME" in features):
                        choices = [e for e in all_entities if e in all_mentions]
                    if len(choices) == 0:
                        choices = all_entities
                    print "Choices (filtered by bot match)", choices
                    entity = np.random.choice(choices)
                    print "Chosen entity:", entity
                else:
                    # no matches!!
                    print "No match features. Choices:", choices
                    bot_entities = get_all_friend_info_with_tag(tag, bot_info)
                    if current_user == bot_idx:
                        all_entities = get_all_friend_info_with_tag(tag, other_user_info)
                    else:
                        all_entities = get_all_friend_info_with_tag(tag, agent_info)

                    print "Bot entities", bot_entities
                    print "Agent entities:", all_entities
                    # make sure you can't match bot info or bot friend info
                    if tag == Entity.to_tag(Entity.MAJOR) and bot_info["info"]["school"]["major"].lower() not in bot_entities:
                        bot_entities.append(bot_info["info"]["school"]["major"].lower())
                    elif tag == Entity.to_tag(Entity.SCHOOL_NAME) and bot_info["info"]["school"]["name"].lower() not in bot_entities:
                        bot_entities.append(bot_info["info"]["school"]["name"].lower())
                    elif tag == Entity.to_tag(Entity.COMPANY_NAME) and bot_info["info"]["company"]["name"].lower() not in bot_entities:
                        bot_entities.append(bot_info["info"]["company"]["name"].lower())
                    choices = [e for e in choices if e in all_entities and e not in bot_entities]
                    if len(choices) == 0 and ("F:MENTIONED_BY_ME" in features or "F:MENTIONED_BY_FRIEND" in features):
                        choices = [e for e in all_entities if e not in bot_entities]
                        if len(choices) == 0:
                            # match feature missed during model generation
                            choices = all_entities

                    entity = np.random.choice(choices)
        new_sentence.append(entity)
        print "New sentence:", new_sentence
        new_mentions[tag].append(entity.lower())

    print "--------------------"
    return " ".join(new_sentence), new_mentions


def parse_transcripts(name, html_lines, replace_entities=False, scenarios=None):
    results_json = json.load(open(name, 'r'))
    ctr = 0
    for result in results_json:

        ctr += 1
        chat_html= ['<h3>Chat_%d</h3>' % ctr, '<table border=\"2\", style=\"border-collapse: collapse\">']
        model_input = [x.replace(SEQ_DELIM, "").strip() for x in result["x"]]
        model_out = result["y_pred"].split(SEQ_DELIM)
        model_ref = result["y_ref"].split(SEQ_DELIM)
        if replace_entities:
            bot_idx = result["agent_idx"]
            scenario_id = result["scenario_id"]
            partner_mentions = []
            partner_mentions_flat = defaultdict(list)
            bot_mentions = []
            bot_mentions_flat = defaultdict(list)

        chat_html.append("<tr><td width=%s><b>Model Input</b></td>" % COL_WIDTH)
        chat_html.append("<td width=%s><b>Tagged Input </b></td>" % COL_WIDTH)
        chat_html.append("<td width=%s><b>Model Output</b></td>" % COL_WIDTH)
        chat_html.append("<td width=%s><b>Tagged Output</b></td>" % COL_WIDTH)

        for i in range(0, len(model_input)):
            m_inp = model_input[i]
            if replace_entities:
                tagged_inp = split_seq_into_sentences(m_inp)
                print "Replacing entities in input..."
                print "Input:", m_inp
                m_inp, new_partner_mentions = replace_entities_in_seq(m_inp, scenarios[scenario_id], bot_idx,1-bot_idx, bot_mentions_flat, partner_mentions_flat)
                if len(partner_mentions) > MENTION_WINDOW:
                    least_recent = partner_mentions.pop(0)
                    for (entity_type, values) in least_recent.iteritems():
                        for value in values:
                            print partner_mentions
                            print entity_type, value
                            print value in partner_mentions_flat[entity_type]
                            print partner_mentions_flat[entity_type]
                            print partner_mentions_flat[entity_type].index(value)
                            print least_recent
                            partner_mentions_flat[entity_type].remove(value)
                partner_mentions.append(new_partner_mentions)
                for (key, values) in new_partner_mentions.iteritems():
                    partner_mentions_flat[key].extend(values)
            inp = split_seq_into_sentences(m_inp)
            m_out = model_out[i]
            if replace_entities:
                print "Replacing entities in model output..."
                print "Output:", m_out
                tagged_out = split_seq_into_sentences(m_out)
                m_out, new_bot_mentions = replace_entities_in_seq(m_out, scenarios[scenario_id], bot_idx, bot_idx, bot_mentions_flat, partner_mentions_flat)
                if len(bot_mentions) > MENTION_WINDOW:
                    least_recent = bot_mentions.pop(0)
                    for(entity_type, values) in least_recent.iteritems():
                        for value in values:
                            bot_mentions_flat[entity_type].remove(value)
                bot_mentions.append(new_bot_mentions)
                for (key, values) in new_bot_mentions.iteritems():
                    bot_mentions_flat[key].extend(values)

            m_out = split_seq_into_sentences(m_out)
            m_ref = split_seq_into_sentences(model_ref[i])

            chat_html.append("</tr><tr>")
            if replace_entities:
                all_lines = [inp, tagged_inp, m_out, tagged_out]
            else:
                all_lines = [inp, m_out, m_ref]

            for (idx, lines) in enumerate(all_lines):
                style = REGULAR_STYLE
                if idx == 2 and not replace_entities:
                    style = OUTPUT_STYLE
                elif replace_entities and idx == 1 or idx == 3:
                    style = OUTPUT_STYLE

                chat_html.append("<td width=%s style=%s> %s" % (COL_WIDTH, style, lines[0]))
                for line in lines[1:]:
                    chat_html.append("<br>%s" % line)
                chat_html.append("</td>")

        chat_html.append("</tr>")
        chat_html.append("</table>")
        html_lines.extend(chat_html)
        # if ctr == 20:
        #     break


def aggregate_chats(args):
    html = ['<!DOCTYPE html>','<html>']
    chats = []
    total = 0
    num_completed = 0
    if args.replace_entities:
        scenarios = load_scenarios(args.scenarios)
        parse_transcripts(args.results, chats, args.replace_entities, scenarios)
    else:
        parse_transcripts(args.results, chats)

    html.extend(chats)
    html.append('</html>')
    return html

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--out', default='reports/', help='Directory to output report to')
    parser.add_argument('--filename', required=True, help='Filename to output report to')
    parser.add_argument('--results', required=True, help='File containing model results')
    parser.add_argument('--replace_entities', type=bool, default=False, help='Flag indicating whether to replace entity tags by entities or not')
    parser.add_argument('--scenarios', type=str, help='File to load scenarios from')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    outfile = codecs.open(os.path.join(args.out, args.filename), 'w', encoding='utf-8')
    html_lines = aggregate_chats(args)

    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()

