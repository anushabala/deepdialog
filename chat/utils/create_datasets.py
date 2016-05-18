__author__ = 'anushabala'
from argparse import ArgumentParser
from transcript_utils import load_scenarios, parse_transcript, is_transcript_valid
import os
import string
import random
import codecs
from chat.modeling.tagger import EntityTagger, Entity

NO_OUTCOME = "NO_OUTCOME"
VALID = "VALID"
SELECT_TEXT = "SELECT NAME"
MAX_NAME_COUNT = 5
MAX_SELECT_COUNT = 2
tagger = None
SEQ_DELIMITER = " | "
MENTION_WINDOW=3


def tag_sequence(seq, scenario, agent_idx=-1, include_features=False, prev_mentions_by_me=None, prev_mentions_by_friend=None):
    if include_features:
        found_entities, possible_entities, features = tagger.tag_sentence(seq, include_features, scenario, agent_idx)
    else:
        found_entities, possible_entities = tagger.tag_sentence(seq)

    all_entities = [found_entities, possible_entities]
    all_matched_tokens = []
    if include_features:
        for entity_dict in all_entities:
            for entity_type in entity_dict.keys():
                all_matched_tokens.extend(entity_dict[entity_type])

        for mentioned in prev_mentions_by_friend:
            if mentioned in all_matched_tokens:
                if "_<F:MENTIONED_BY_FRIEND>" not in features[mentioned]:
                    features[mentioned] += "_<F:MENTIONED_BY_FRIEND>"
            elif " " in mentioned:
                split = mentioned.split()
                for part in split:
                    if part in all_matched_tokens:
                        if "_<F:MENTIONED_BY_FRIEND>" not in features[part]:
                            features[part] += "_<F:MENTIONED_BY_FRIEND>"
            else:
                for token in all_matched_tokens:
                    if " " in token:
                        split = token.split()
                        # if "geissler" in token:

                        for part in split:
                            if part == mentioned:
                                if "_<F:MENTIONED_BY_FRIEND>" not in features[token]:
                                    features[token] += "_<F:MENTIONED_BY_FRIEND>"

        for mentioned in prev_mentions_by_me:
            if mentioned in all_matched_tokens:
                if "_<F:MENTIONED_BY_ME>" not in features[mentioned]:
                    features[mentioned] += "_<F:MENTIONED_BY_ME>"
            elif " " in mentioned:
                split = mentioned.split()
                for part in split:
                    if part in all_matched_tokens:
                        if "_<F:MENTIONED_BY_ME>" not in features[part]:
                            features[part] += "_<F:MENTIONED_BY_ME>"
            else:
                for token in all_matched_tokens:
                    if " " in token:
                        # if "geissler" in token:
                        #     print all_matched_tokens
                        #     print mentioned
                        split = token.split()
                        for part in split:
                            if part in mentioned:
                                if "_<F:MENTIONED_BY_ME>" not in features[token]:
                                    features[token] += "_<F:MENTIONED_BY_ME>"

    sentence = seq.strip().split()
    # print sentence
    new_sentence = []
    for entity_type in Entity.types():
        for entity_dict in all_entities:
            if entity_type not in entity_dict.keys():
                continue
            matched_tokens = entity_dict[entity_type]
            # if entity_type == Entity.SCHOOL_NAME:
            #     print matched_tokens
            for token in matched_tokens:
                i = 0
                while i < len(sentence):
                    if " " in token:
                        split_token = [t.strip() for t in token.split()]
                        try:
                            sentence_tokens = [w.strip() for w in sentence[i:i+len(split_token)]]
                            if split_token == sentence_tokens:
                                if include_features:
                                    new_sentence.append(features[token])
                                else:
                                    new_sentence.append("<%s>" % Entity.to_tag(entity_type))
                                # if "krone" in split_token:
                                #     print "SENTENCE", sentence
                                #     print split_token
                                #     print sentence_tokens, i
                                #     print new_sentence
                                i += len(split_token)
                            else:
                                new_sentence.append(sentence[i])
                                i += 1
                        except IndexError:
                            new_sentence.append(sentence[i])
                            i+=1
                    elif token == sentence[i]:
                        if include_features:
                            new_sentence.append(features[token])
                        else:
                            new_sentence.append("<%s>" % Entity.to_tag(entity_type))
                        i+=1
                    else:
                        new_sentence.append(sentence[i])
                        i+=1
                sentence = new_sentence
                # print sentence
                new_sentence = []

    sentence = " ".join(sentence)
    # print sentence
    # for entity_type in Entity.types():
    #     for entity_dict in all_entities:
    #         matched_tokens = entity_dict[entity_type]
    #         for token in matched_tokens:
    #             if " " in token:
    #                 sentence = sentence.replace(token, Entity.to_tag(entity_type))
    if include_features:
        return sentence, all_matched_tokens
    return sentence


def construct_metadata_string(scenario, scenarios, agent_num):
    agent = scenarios[scenario]["agents"][agent_num]
    agent_info = agent["info"]
    metadata = "[ ME NAME %s SCHOOL %s MAJOR %s COMPANY %s ]" % \
              (agent_info["name"],
               agent_info["school"]["name"],
               agent_info["school"]["major"],
               agent_info["company"]["name"])
    for friend in agent["friends"]:
        metadata += " [ FRIEND NAME %s SCHOOL %s MAJOR %s COMPANY %s ]" % \
                    (friend["name"],
                     friend["school"]["name"],
                     friend["school"]["major"],
                     friend["company"]["name"])

    return metadata


def get_sequences_from_transcript(transcript, scenarios, reverse=False, include_metadata=False,
                                  tag_sentences=False, include_features=False):
    dialogue = transcript['dialogue']
    current_user = -2
    first_user = -2
    other_user = -2
    bot_user = -2
    seq_in = []
    seq_out = []
    prev_mentions = {}
    if reverse:
        seq_out.append("START")
    current_seq = seq_in
    current_text = ""
    if not reverse:
        current_text = "START"
    for (user_num, text) in dialogue:
        text = text.strip()
        text = text.translate(string.maketrans("",""), string.punctuation)
        if current_user < -1:
            current_user = user_num
            first_user = user_num
            prev_mentions[first_user] = []
            other_user = 1 - first_user
            prev_mentions[other_user] = []
            bot_user = other_user
            if reverse:
                bot_user = first_user


        if user_num != current_user:
            text_to_add = current_text.strip()
            if tag_sentences:
                print text_to_add
                if include_features:
                    my_mentions_flattened = []
                    for mentions in prev_mentions[current_user]:
                        my_mentions_flattened.extend(mentions)
                    friend_mentions_flattened = []
                    for mentions in prev_mentions[1 - current_user]:
                        friend_mentions_flattened.extend(mentions)
                    text_to_add, new_mentions = tag_sequence(text_to_add, scenarios[transcript["scenario"]], bot_user, include_features=include_features, prev_mentions_by_me=my_mentions_flattened, prev_mentions_by_friend=friend_mentions_flattened)
                    if len(prev_mentions[current_user]) > MENTION_WINDOW:
                        prev_mentions[current_user].pop(0)
                    prev_mentions[current_user].append(new_mentions)
                else:
                    text_to_add = tag_sequence(text_to_add, scenarios[transcript["scenario"]], bot_user)
                print text_to_add
                # print "------"
            current_seq.append(text_to_add)
            current_text = ""
            current_seq = seq_out if current_seq == seq_in else seq_in
            current_user = user_num

        if SELECT_TEXT in text:
            current_text += " " + text
        else:
            current_text += " SAY " + text.lower()

    text_to_add = current_text.strip()
    if tag_sentences:
        print text_to_add
        if include_features:
            my_mentions_flattened = []
            for mentions in prev_mentions[current_user]:
                my_mentions_flattened.extend(mentions)
            friend_mentions_flattened = []
            for mentions in prev_mentions[1 - current_user]:
                friend_mentions_flattened.extend(mentions)
            text_to_add, new_mentions = tag_sequence(text_to_add, scenarios[transcript["scenario"]], bot_user, include_features=include_features, prev_mentions_by_me=my_mentions_flattened, prev_mentions_by_friend=friend_mentions_flattened)
            if len(prev_mentions[current_user]) > MENTION_WINDOW:
                prev_mentions[current_user].pop(0)
            prev_mentions[current_user].append(new_mentions)
        else:
            text_to_add = tag_sequence(text_to_add, scenarios[transcript["scenario"]], bot_user)
        print text_to_add
    current_seq.append(text_to_add)
    if len(seq_in) < len(seq_out):
        seq_in.append("PASS")
    if len(seq_in) > len(seq_out):
        seq_out.append("PASS")

    try:
        assert len(seq_in) == len(seq_out)
    except AssertionError:
        print dialogue
        print seq_in
        print seq_out
        exit(1)

    if reverse:
        metadata = construct_metadata_string(transcript["scenario"], scenarios, other_user)
        if include_metadata:
            seq_out[0] = metadata + " " + seq_out[0]
        return SEQ_DELIMITER.join(seq_out).strip(), SEQ_DELIMITER.join(seq_in).strip(), other_user, transcript["scenario"]
    else:
        metadata = construct_metadata_string(transcript["scenario"], scenarios, first_user)
        if include_metadata:
            seq_in[0] = metadata + " " + seq_in[0]
        return SEQ_DELIMITER.join(seq_in).strip(), SEQ_DELIMITER.join(seq_out).strip(), first_user, transcript["scenario"]


def is_degenerate(seq):
    sentences = seq.split(SEQ_DELIMITER)
    for s in sentences:
        if s.count(Entity.to_tag(Entity.FULL_NAME)) > MAX_NAME_COUNT or s.count(SELECT_TEXT) > MAX_SELECT_COUNT:
            return True
    return False

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, default='../scenarios.json', help='File containing JSON scenarios')
    parser.add_argument("--transcripts", type=str, default='../transcripts', help='Directory containing chat transcripts')
    parser.add_argument("--out_dir", type=str, default='../transcripts_with_prefs', help='Directory to write output transcripts to')
    parser.add_argument('--prefix', type=str, default='friends.seq2seq.rel.aligned')
    parser.add_argument('--include_metadata', action='store_true')
    parser.add_argument('--tag_entities', action='store_true')
    parser.add_argument('--include_features', action='store_true')

    args = parser.parse_args()
    scenarios = load_scenarios(args.scenarios)
    out_dir = args.out_dir
    prefix = args.prefix

    tagger = EntityTagger(scenarios)

    train_out = codecs.open(os.path.join(out_dir, prefix)+'.train', 'a', encoding='utf-8')
    val_out = codecs.open(os.path.join(out_dir, prefix)+'.val', 'a', encoding='utf-8')
    test_out = codecs.open(os.path.join(out_dir, prefix)+'.test', 'a', encoding='utf-8')
    if args.include_metadata or args.tag_entities:
        train_ids = open(os.path.join(out_dir, prefix)+'.train_ids', 'a')
        val_ids = open(os.path.join(out_dir, prefix)+'.val_ids', 'a')
        test_ids = open(os.path.join(out_dir, prefix)+'.test_ids', 'a')
    degenerate_ctr = 0
    invalid = {NO_OUTCOME: 0, VALID: 0}

    for name in os.listdir(args.transcripts):
        f = os.path.join(args.transcripts, name)
        transcript = parse_transcript(f, include_bots=False)
        if transcript is None:
            continue
        valid, reason = is_transcript_valid(transcript)
        if not valid:
            invalid[reason] += 1
        else:
            invalid[VALID] += 1
            # print transcript
            in_seq, out_seq, agent_idx, scenario_id = get_sequences_from_transcript(transcript,
                                                            scenarios,
                                                            include_metadata=args.include_metadata,
                                                            tag_sentences=args.tag_entities,
                                                            include_features=args.include_features)
            in_rev, out_rev, agent_idx_rev, scenario_id_rev = get_sequences_from_transcript(transcript,
                                                            scenarios,
                                                            reverse=True,
                                                            include_metadata=args.include_metadata,
                                                            tag_sentences=args.tag_entities,
                                                            include_features=args.include_features)

            if is_degenerate(in_seq) or is_degenerate(out_seq):
                # print "degenerate example, skipping", transcript
                # print in_seq, out_seq
                # print "---"
                degenerate_ctr += 1
                continue
            r = random.random()
            if 0 <= r < 0.6:
                train_out.write("%s\t%s\t%d\t%s\n" % (in_seq, out_seq, agent_idx, scenario_id))
                train_out.write("%s\t%s\t%d\t%s\n" % (in_rev, out_rev, agent_idx_rev, scenario_id_rev))
                if args.include_metadata or args.tag_entities:
                    train_ids.write("%s\n" % transcript["scenario"])
                    train_ids.write("%s\n" % transcript["scenario"])
            elif 0.6 <= r < 0.8:
                val_out.write("%s\t%s\t%d\t%s\n" % (in_seq, out_seq, agent_idx, scenario_id))
                val_out.write("%s\t%s\t%d\t%s\n" % (in_rev, out_rev, agent_idx_rev, scenario_id_rev))
                if args.include_metadata or args.tag_entities:
                    val_ids.write("%s\n" % transcript["scenario"])
                    val_ids.write("%s\n" % transcript["scenario"])
            else:
                test_out.write("%s\t%s\t%d\t%s\n" % (in_seq, out_seq, agent_idx, scenario_id))
                test_out.write("%s\t%s\t%d\t%s\n" % (in_rev, out_rev, agent_idx_rev, scenario_id_rev))
                if args.include_metadata or args.tag_entities:
                    test_ids.write("%s\n" % transcript["scenario"])
                    test_ids.write("%s\n" % transcript["scenario"])

    train_out.close()
    val_out.close()
    test_out.close()

    if args.include_metadata:
        train_ids.close()
        val_ids.close()
        test_ids.close()

    print "Total number of degenerate examples: %d of %d valid examples" % (degenerate_ctr, invalid[VALID])