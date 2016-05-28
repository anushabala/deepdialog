import json

# Global statistics that we can output to monitor the run.

stats_path = None
STATS = {}

def init(path):
    global stats_path
    stats_path = path

def add(*args):
    # Example: add_stats('data', 'num_examples', 3)
    s = STATS
    prefix = args[:-2]
    for k in prefix:
        if k not in s:
            s[k] = {}
        s = s[k]
    s[args[-2]] = args[-1]

    if stats_path:
        out = open(stats_path, 'w')
        print >>out, json.dumps(STATS)
        out.close()
