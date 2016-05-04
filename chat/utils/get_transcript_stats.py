__author__ = 'anushabala'

from argparse import ArgumentParser
import os
import shutil
import codecs
from transcript_utils import parse_transcript, is_transcript_short, is_transcript_valid, load_scenarios

NO_OUTCOME = "NO_OUTCOME"
TOO_SHORT = "TOO_SHORT"
SHORT = "SHORT"
BOTH = "BOTH"
NEITHER = "NEITHER"
ONE = "ONE"
VALID = "VALID"


def user_picked_optimal(user_num, transcript, scenarios):
    scenario = scenarios[transcript["scenario"]]
    idx, name = transcript["choices"][user_num]
    sorted_restaurants = scenario["agents"][user_num]["sorted_restaurants"]
    max_utility = sorted_restaurants[0]["utility"]
    try:
        choice_utility = next(obj["utility"] for obj in scenario["agents"][user_num]["sorted_restaurants"] if
                              obj["name"] == name)
    except StopIteration:
        choice_utility = 0
    if choice_utility == max_utility or sorted_restaurants[2]["name"] == name:
        return True

    return False


def write_preferences(open_file, scenario):
    user0_prefs = scenario["agents"][0]
    user1_prefs = scenario["agents"][1]
    open_file.write("User 0 preferences: %s\t%s\n" %
                    ("-".join([str(x) for x in user0_prefs["spending_func"][0]["price_range"]]),
                    ", ".join([c["cuisine"] for c in user0_prefs["cuisine_func"]])))
    open_file.write("User 1 preferences: %s\t%s\n" %
                    ("-".join([str(x) for x in user1_prefs["spending_func"][0]["price_range"]]),
                    ", ".join([c["cuisine"] for c in user1_prefs["cuisine_func"]])))


def write_available_restaurants(open_file, scenario):
    restaurants = scenario["restaurants"]
    for r in restaurants:
        open_file.write("%s\t%s\t%s\n" % (r["name"], r["cuisine"], "-".join([str(x) for x in r["price_range"]])))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, default='../scenarios.json', help='File containing JSON scenarios')
    parser.add_argument("--transcripts", type=str, default='../transcripts', help='Directory containing chat transcripts')
    parser.add_argument("--out_dir", type=str, default='../transcripts_with_prefs', help='Directory to write output transcripts to')
    args = parser.parse_args()
    scenarios = load_scenarios(args.scenarios)
    out_dir = args.out_dir

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    ctr = 0
    invalid = {NO_OUTCOME: 0, TOO_SHORT: 0, SHORT: 0, VALID: 0}
    optimal_choice = {BOTH: 0, NEITHER: 0, ONE: 0}

    for name in os.listdir(args.transcripts):
        f = os.path.join(args.transcripts, name)
        transcript = parse_transcript(f)
        if transcript is None:
            continue
        valid, reason = is_transcript_valid(transcript)
        if not valid:
            invalid[reason] += 1
        else:
            out = os.path.join(out_dir, name)
            shutil.copyfile(f, out)
            out_file = codecs.open(out, mode='a', encoding='utf-8')
            out_file.write("\nChat Information:\n\n")
            write_preferences(out_file, scenarios[transcript["scenario"]])
            out_file.write("\n")
            write_available_restaurants(out_file, scenarios[transcript["scenario"]])
            out_file.write("\n")

            short, reason = is_transcript_short(transcript)
            if short:
                invalid[reason] += 1
            else:
                invalid[VALID] += 1

            user_0_optimal = user_picked_optimal(0, transcript, scenarios)
            user_1_optimal = user_picked_optimal(1, transcript, scenarios)

            if user_0_optimal and user_1_optimal:
                optimal_choice[BOTH] += 1
            elif user_0_optimal or user_1_optimal:
                optimal_choice[ONE] += 1
            else:
                optimal_choice[NEITHER] += 1

            out_file.write("Final selection: %s" % transcript["outcome"][1])
            if user_0_optimal:
                out_file.write("User 0 selected their optimal choice (or something close).\n")
            else:
                out_file.write("User 0 did not select their optimal choice.")
            if user_1_optimal:
                out_file.write("User 1 selected their optimal choice (or something close).\n")
            else:
                out_file.write("User 1 did not select their optimal choice.")

            out_file.close()
        ctr += 1



    print optimal_choice
    print invalid
