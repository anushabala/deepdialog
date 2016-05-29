There are two steps:

- Read the raw transcripts and output a sequence-to-sequence dataset.
- Train a sequence-to-sequence model.

# Running on CodaLab

    cl work nlp::pliang-dialog

    # Upload data and code
    cl upload chat/backups_from_remote/transcripts_0520_friends
    cl upload chat -x backups_from_remote

    # Old tagging
    cl run :chat data:transcripts_0520_friends 'PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friends_scenarios.json --transcripts data --out_dir . --prefix data --tag_entities --include_features' --name tagged

    # New tagging
    cl run :chat raw:transcripts_0520_friends 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios chat/friends_scenarios.json --transcripts raw --out-prefix tagged' --name tagged

    # Train a model
    cl upload chat -x backups_from_remote; cl run :chat :tagged 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --train-data tagged/data.train --dev-data tagged/data.val --out-dir .'

# Running locally

    # Old tagging
    mkdir -p output/friends
    PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friends_scenarios.json --transcripts chat/backups_from_remote/transcripts_0520_friends --out_dir output/friends --prefix 0520_friends --tag_entities --include_features

    # New JSON format
    mkdir -p output/friends
    PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios chat/friends_scenarios.json --transcripts chat/backups_from_remote/transcripts_0520_friends --out-prefix output/friends/0520_friends

    # Visualize
    cat output/friends/0520_friends.train.json | jq . | less
    cat output/friends/0520_friends.train.json | jq -r '.[].seqs[].seq[].formula_tokens[]' | sort | uniq -c | sort -nr | less
    cat output/friends/0520_friends.entity_phrase.json | jq . | less

    # Train a model
    mkdir -p output/friends/0520_friends.run
    PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --data-prefix output/friends/0520_friends --out-dir output/friends/0520_friends.run

# Setting up the webserver

    virtualenv venv
    venv/bin/pip install -r chat/requirements.txt
    PYTHONPATH=. venv/bin/python chat/start_app.py -p chat/params.json
