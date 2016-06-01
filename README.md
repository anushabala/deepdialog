There are two steps:

- Read the raw transcripts and output a sequence-to-sequence dataset.
- Train a sequence-to-sequence model.

# Dataset JSON format

Each example in the dataset specifies a proposal distribution over possible sequences.

    Let X be the top-level JSON object.
    X is a list of examples.
    X[i]['states'] is a list of states (possible parses of the raw tokens into entities).
    X[i]['states'][j].weight: sample this state with probability proportional to this weight.
    X[i]['states'][j].messages: list of messages (one person talking)
    X[i]['states'][j].messages[k]['formula_token_candidates'][p]: is a list of candidates for the p-th position
    X[i]['states'][j].messages[k]['formula_token_candidates'][p][l]: the l-th candidate, which is a token-weight pair

The distribution is gotten by:

- Sampling a state according to the given probabilities.
- For each token, sample a candidate token independently with probability proportional to its weight.

# Running on CodaLab

    cl work nlp::pliang-dialog

    # Upload data
    cl upload data/backups_from_remote/transcripts_0520_friends
    cl upload data/backups_from_remote/transcripts_0523_friends
    cl upload data/backups_from_remote/transcripts_0524_dating
    cl upload data/friends_scenarios.json
    cl upload data/matchmaking_scenarios.json

    # Upload code
    cl upload chat
    cl mimic chat^2 chat  # Rerun everything

    # Generate datasets: raw transcripts => JSON logical forms dataset
    cl run :chat raw1:transcripts_0520_friends raw2:transcripts_0523_friends scenarios.json:friends_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw1 raw2 --out-prefix ./' --name friends.tagged
    cl run :chat raw:transcripts_0524_dating scenarios.json:matchmaking_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw --out-prefix ./' --name matchmaking.tagged
    cl download matchmaking.tagged/tagged.train.json -o output/0524_dating.train.json

# Running locally

    # Generate datasets
    mkdir -p output
    PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios data/friends_scenarios.json --transcripts data/backups_from_remote/transcripts_0520_friends --out-prefix output/0520_friends.
    PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios data/matchmaking_scenarios.json --transcripts data/backups_from_remote/transcripts_0524_dating --out-prefix output/0524_dating.

    # Try out the dialogue manager with simple bigram model
    PYTHONPATH=. python chat/utils/dialogue_main.py --scenarios data/friends_scenarios.json --train output/0520_friends.train.json

    # Visualize
    cat output/0520_friends.train.json | jq . | less
    cat output/0520_friends.train.json | jq -r '.[].seqs[].seq[].formula_tokens[]' | sort | uniq -c | sort -nr | less
    cat output/0520_friends.entity_phrase.json | jq . | less
    cat output/0524_dating.train.json | jq -r '.[].seqs[].seq[].formula_tokens[]' | sort | uniq -c | sort -nr | less
    cat output/0524_dating.entity_phrase.json | jq . | less

    # Train a model
    PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --data-prefix output/0520_friends --out-dir output/0520_friends.run

# Old commands (deprecated)

Running on CodaLab:

    # Tagging
    cl run :chat data:transcripts_0520_friends 'PYTHONPATH=. python chat/utils/create_datasets.py --scenarios chat/friends_scenarios.json --transcripts data --out_dir . --prefix data --tag_entities --include_features' --name tagged

    # Train a model
    cl run :chat :tagged 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --train-data tagged/data.train --dev-data tagged/data.val --out-dir .'

Running locally:

    # Tagging
    mkdir -p output/friends
    PYTHONPATH=. python chat/utils/create_datasets.py --scenarios data/friends_scenarios.json --transcripts data/backups_from_remote/transcripts_0520_friends --out_dir output --prefix 0520_friends_old --tag_entities --include_features

    # Train a model
    mkdir -p output/friends/0520_friends.run
    PYTHONPATH=. python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --data-prefix output/0520_friends_old --out-dir output/0520_friends_old.run


# Setting up the webserver

    virtualenv venv
    venv/bin/pip install -r chat/requirements.txt
    PYTHONPATH=. venv/bin/python chat/start_app.py -p chat/params.json
