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

    # Upload raw data
    cl work main::pliang-dialog-data
    cl upload data/backups_from_remote/transcripts_0520_friends
    cl upload data/backups_from_remote/transcripts_0523_friends
    cl upload data/backups_from_remote/transcripts_0524_dating
    cl upload data/friends_scenarios.json
    cl upload data/matchmaking_scenarios.json

    # Generate datasets: raw transcripts => JSON logical forms dataset
    cl work main::pliang-dialog-data
    cl upload chat
    cl run :chat raw1:transcripts_0520_friends raw2:transcripts_0523_friends scenarios.json:friends_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw1 raw2 --out-prefix ./ --formulas-mode verbatim' --name friends.verbatim --request-network
    cl run :chat raw1:transcripts_0520_friends raw2:transcripts_0523_friends scenarios.json:friends_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw1 raw2 --out-prefix ./ --formulas-mode basic' --name friends.basic --request-network
    cl run :chat raw1:transcripts_0520_friends raw2:transcripts_0523_friends scenarios.json:friends_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw1 raw2 --out-prefix ./ --formulas-mode recurse' --name friends.recurse --request-network
    cl run :chat raw:transcripts_0524_dating scenarios.json:matchmaking_scenarios.json 'PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios scenarios.json --transcripts raw --out-prefix ./' --name matchmaking.tagged --request-network

    # Download datasets (optional)
    cl download matchmaking.tagged/tagged.train.json -o output/0524_dating.train.json

    # Train a model
    cl work main::pliang-dialog2
    cl upload chat
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.verbatim 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 15 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --train-eval-period 10 --formulas-mode verbatim' -n friends.run --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.basic 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 15 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --train-eval-period 10 --formulas-mode basic' -n friends.run --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.recurse 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 15 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --train-eval-period 10 --formulas-mode recurse' -n friends.run --request-network
    # try numsamples
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.recurse 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 10 --batch-size 5 --num-samples 5 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --train-eval-period 10 --formulas-mode recurse' -n friends.run --request-network
    # small
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.verbatim 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 100 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --formulas-mode verbatim --train-max-examples 5 --dev-max-examples 0' -n friends.run --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.basic 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 100 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --formulas-mode basic --train-max-examples 5 --dev-max-examples 0' -n friends.run --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.recurse 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 100 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --formulas-mode recurse --train-max-examples 5 --dev-max-examples 0' -n friends.run --request-network
    cl upload chat
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.recurse 'THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True,blas.ldflags=-lopenblas PYTHONPATH=. python chat/nn/main.py -d 100 -i 50 -o 50 -t 100 --batch-size 5 --num-samples 1 --scenarios scenarios.json --data-prefix tagged/ --out-dir . --formulas-mode recurse --train-max-examples 5 --dev-max-examples 5' -n friends.run --request-network

    # Train simple n-gram model
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.recurse 'PYTHONPATH=. python chat/utils/dialogue_main.py --formulas-mode recurse --scenarios scenarios.json --train tagged/train.json --dev tagged/dev.json --stats-file stats.json --n 13' -n friends.ngram --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.basic 'PYTHONPATH=. python chat/utils/dialogue_main.py --formulas-mode basic --scenarios scenarios.json --train tagged/train.json --dev tagged/dev.json --stats-file stats.json --n 13' -n friends.ngram --request-network
    cl run :chat scenarios.json:friends_scenarios.json tagged:friends.verbatim 'PYTHONPATH=. python chat/utils/dialogue_main.py --formulas-mode verbatim --scenarios scenarios.json --train tagged/train.json --dev tagged/dev.json --stats-file stats.json --n 13' -n friends.ngram --request-network

# Running locally

    # Generate datasets
    mkdir -p output
    PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios data/friends_scenarios.json --transcripts data/backups_from_remote/transcripts_0520_friends --out-prefix output/0520_friends.
    PYTHONPATH=. python chat/utils/create_json_dataset.py --scenarios data/matchmaking_scenarios.json --transcripts data/backups_from_remote/transcripts_0524_dating --out-prefix output/0524_dating.

    # Try out the dialogue manager with simple n-gram model
    PYTHONPATH=. python chat/utils/dialogue_main.py --formulas-mode recurse --scenarios data/friends_scenarios.json --train output/0520_friends.train.json --dev output/0520_friends.dev.json --n 4

    # Visualize
    cat output/0520_friends.train.json | jq . | less
    cat output/0520_friends.train.json | jq -r '.[].seqs[].seq[].formula_tokens[]' | sort | uniq -c | sort -nr | less
    cat output/0520_friends.entity_phrase.json | jq . | less
    cat output/0524_dating.train.json | jq -r '.[].seqs[].seq[].formula_tokens[]' | sort | uniq -c | sort -nr | less
    cat output/0524_dating.entity_phrase.json | jq . | less

    # Train a model
    PYTHONPATH=. venv/bin/python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --data-prefix output/0520_friends --out-dir output/0520_friends.run

    # Train small model
    PYTHONPATH=. venv/bin/python chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 --num-samples 5 --data-prefix output/0520_friends. --out-dir output/0520_friends.run --train-max-examples 1 --dev-max-examples 0 --scenarios data/friends_scenarios.json --num-epochs 10

# Setting up the webserver

    virtualenv venv
    venv/bin/pip install -r chat/requirements.txt
    PYTHONPATH=. venv/bin/python chat/start_app.py -p chat/params.json
