# Tagging utterances

This command reads the raw transcripts and outputs tagged utterances for training:

    mkdir -p output/friends
    PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friend_scenarios.json --transcripts chat/backups_from_remote/transcripts_0520_friends --out_dir output/friends --prefix 0520-friends --tag_entities --include_features

# How to train
