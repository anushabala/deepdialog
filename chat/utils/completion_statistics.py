__author__ = 'anushabala'
from argparse import ArgumentParser
import os
from transcript_utils import parse_transcript

LSTM_UNFEATURIZED = "LSTM_UNFEATURIZED"
LSTM_FEATURIZED = "LSTM_FEATURIZED"
BASELINE = "DEFAULT_BOT"
HUMAN = "human"
NO_OUTCOME = "NO_OUTCOME"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--transcripts', type=str, nargs='+', help='Directories where transcripts are located')
    args = parser.parse_args()

    total_attempts = {HUMAN:0.0, BASELINE:0.0, LSTM_FEATURIZED:0.0, LSTM_UNFEATURIZED:0.0}
    complete_attempts = {HUMAN:0.0, BASELINE:0.0, LSTM_FEATURIZED:0.0, LSTM_UNFEATURIZED:0.0}
    for dir in args.transcripts:
        print dir
        for name in os.listdir(dir):
            # print name
            file = os.path.join(dir, name)
            transcript = parse_transcript(file, True)
            if "BOT_TYPE" in transcript.keys() and transcript["BOT_TYPE"] is not None:
                total_attempts[transcript["BOT_TYPE"]] += 1

                if transcript["outcome"] != NO_OUTCOME:
                    complete_attempts[transcript["BOT_TYPE"]] += 1
    
    for key in total_attempts.keys():
        print "%s" % key.title()
        print "Total attempts: %d" % total_attempts[key]
        print "Completed chats: %d" % complete_attempts[key]
        print "%% chats completed: %2.2f" % (complete_attempts[key]/total_attempts[key])
