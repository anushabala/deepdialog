# CodaLab

    cl work nlp::pliang-dialog
    cl upload chat/backups_from_remote/transcripts_0520_friends
    # Tag
    cl run :chat data:transcripts_0520_friends 'PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friends_scenarios.json --transcripts data --out_dir . --prefix data --tag_entities --include_features' --name tagged
    # Train
    cl upload chat -x backups_from_remote; cl run :chat :tagged 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --train-data tagged/data.train --dev-data tagged/data.val --out-dir .'

# Tagging utterances

This command reads the raw transcripts and outputs tagged utterances for training:

    mkdir -p output/friends
    PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friend_scenarios.json --transcripts chat/backups_from_remote/transcripts_0520_friends --out_dir output/friends --prefix 0520_friends --tag_entities --include_features

# How to train

    chat/trainTaggedLSTM.sh output/friends 0520_friends run1
