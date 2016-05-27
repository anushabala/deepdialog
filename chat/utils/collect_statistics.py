import os

__author__ = 'anushabala'

import sqlite3
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pylab

LSTM_UNFEATURIZED = "LSTM_UNFEATURIZED"
LSTM_FEATURIZED = "LSTM_FEATURIZED"
BASELINE = "DEFAULT_BOT"
HUMAN = "human"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--db', type=str, nargs='+', help='Databases to get information from for')
    parser.add_argument('--out_dir', type=str, default='statistics', help='Directory to output graphs to')
    parser.add_argument('--name', type=str, default='friends', help='Prefix to identify this batch of data')

    args = parser.parse_args()
    databases = args.db
    effectiveness = {LSTM_FEATURIZED:[], LSTM_UNFEATURIZED:[], HUMAN:[], BASELINE:[]}
    human_like = {LSTM_FEATURIZED:[], LSTM_UNFEATURIZED:[], HUMAN:[], BASELINE:[]}
    for db in databases:
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute('SELECT * FROM Surveys')
        x = c.fetchall()
        for item in x:
            human_like[item[1]].append(item[2])
            effectiveness[item[1]].append(item[3])
        conn.close()

    print "%s: Total surveys: %d" % (HUMAN, len(human_like[HUMAN]))
    print "Human-like: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(human_like[HUMAN]), np.std(human_like[HUMAN]))
    print "Effectiveness: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(effectiveness[HUMAN]), np.std(effectiveness[HUMAN]))

    print "%s: Total surveys: %d" % (BASELINE, len(human_like[BASELINE]))
    print "Human-like: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(human_like[BASELINE]), np.std(human_like[BASELINE]))
    print "Effectiveness: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(effectiveness[BASELINE]), np.std(effectiveness[BASELINE]))

    print "%s: Total surveys: %d" % (LSTM_UNFEATURIZED, len(human_like[LSTM_UNFEATURIZED]))
    print "Human-like: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(human_like[LSTM_UNFEATURIZED]), np.std(human_like[LSTM_UNFEATURIZED]))
    print "Effectiveness: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(effectiveness[LSTM_UNFEATURIZED]), np.std(effectiveness[LSTM_UNFEATURIZED]))

    print "%s: Total surveys: %d" % (LSTM_FEATURIZED, len(human_like[LSTM_FEATURIZED]))
    print "Human-like: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(human_like[LSTM_FEATURIZED]), np.std(human_like[LSTM_FEATURIZED]))
    print "Effectiveness: Average: %2.4f\tStandard deviation: %2.4f" % (np.mean(effectiveness[LSTM_FEATURIZED]), np.std(effectiveness[LSTM_FEATURIZED]))

    labels = ["human-like rating", "effectiveness rating"]
    n_bins = 5
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flat

    ax0.hist([human_like[HUMAN], effectiveness[HUMAN]], n_bins, label=labels)
    ax0.set_title('Ratings for humans')
    ax0.legend(loc='best')

    ax1.hist([human_like[BASELINE], effectiveness[BASELINE]], n_bins, label=labels)
    ax1.set_title('Ratings for baseline')
    ax1.legend(loc='best')

    ax2.hist([human_like[LSTM_UNFEATURIZED], effectiveness[LSTM_UNFEATURIZED]], n_bins, label=labels)
    ax2.set_title('Ratings for unfeaturized LSTM')
    ax2.legend(loc='best')

    ax3.hist([human_like[LSTM_FEATURIZED], effectiveness[LSTM_FEATURIZED]], n_bins, label=labels)
    ax3.set_title('Ratings for featurized LSTM')
    ax3.legend(loc='best')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(args.out_dir, args.name)+'.png')
    plt.show()
    plt.close()
