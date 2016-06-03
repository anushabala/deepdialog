#!/bin/env python
import codecs
from collections import defaultdict
from app import create_app, socketio
import sqlite3
import os
import shutil
import json
from argparse import ArgumentParser
from chat.app.main import backend
from chat.modeling.lexicon import Lexicon
from chat.nn.neural import NeuralBox
from nn import spec as specutil
from nn.encoderdecoder import EncoderDecoderModel
from utils.dialogue_main import learn_bigram_model
from modeling.recurrent_box import BigramBox
from modeling.dialogue_tracker import add_arguments


def init_database(db_file, bot_probabilities={}):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # number: room number, participants: number of participants (0 - 2)
    c.execute('''CREATE TABLE ActiveUsers (name text unique, status integer, status_timestamp integer, connected_status integer, connected_timestamp integer, message text, room_id integer, partner_id text, scenario_id text, agent_index integer, selected_index integer, single_task_id text, num_single_tasks_completed integer, num_chats_completed integer, cumulative_points integer, bonus integer)''')
    c.execute('''CREATE TABLE SingleTasks (name text, scenario_id text, selected_index integer, selected_restaurant text, start_text text)''')
    c.execute('''CREATE TABLE CompletedTasks (name text, mturk_code text, num_single_tasks_completed integer, num_chats_completed integer, bonus integer)''')
    c.execute('''CREATE TABLE Surveys (name text, partner_type text, how_mechanical integer, how_effective integer)''')

    c.execute('''CREATE TABLE ChatCounts (id text unique, count integer, prob float)''')
    for bot_name in bot_probabilities.keys():
        c.execute('''INSERT INTO ChatCounts VALUES (?,1,?)''', (bot_name,bot_probabilities[backend.Partner.Human]))

    conn.commit()
    conn.close()


def clear_data(logging_dir):
    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)
    os.makedirs(logging_dir)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', help="File containing app configuration params", type=str,
                        default="params.json")
    parser.add_argument('--host', help="Host IP address to run app on - defaults to localhost", type=str, default="127.0.0.1")
    parser.add_argument('--port', help="Port to run app on", type=int, default=5000)
    parser.add_argument('--log', help="File to log app output to", type=str, default="chat.log")
    add_arguments(parser)
    args = parser.parse_args()
    params_file = args.p
    with open(params_file) as fin:
       params = json.load(fin)

    if os.path.exists(params["db"]["location"]):
        os.remove(params["db"]["location"])

    clear_data(params["logging"]["chat_dir"])
    templates_dir = params["templates_dir"]
    app = create_app(debug=True, templates_dir=templates_dir)

    with codecs.open(params["scenarios_json_file"]) as fin:
        scenarios = json.load(fin)
        scenarios_dict = {v["uuid"]:v for v in scenarios}

    app.config["user_params"] = params
    app.config["scenarios"] = scenarios_dict
    app.config["outcomes"] = defaultdict(lambda : -1)
    app.config["paired_bots"] = defaultdict(None)
    app.config["bot_selections"] = defaultdict(None)
    app.config["lexicon"] = Lexicon(scenarios_dict)

    app.config["bot_waiting_probability"] = {}
    app.config["bot_probability"] = {}

    model_names = []
    models = {}
    for (bot_name, info) in params["bots"].iteritems():
        if info["use_bot"]:
            model_names.append(bot_name)
            model_path = info["path"]
            if bot_name == "bigram":
                cpt = learn_bigram_model(json.load(open(model_path, 'r')))
                box = BigramBox(cpt)
            elif bot_name == "baseline":
                # todo baselinebox?
                cpt = learn_bigram_model(json.load(open(model_path, 'r')))
                box = BigramBox(cpt)
            else:
                spec = specutil.load(model_path)
                model = EncoderDecoderModel(args, spec)
                box = NeuralBox(model)

            models[bot_name] = box

    num_bots = len(model_names)
    bot_waiting_probability = bot_probability = 1.0/(num_bots+1) if num_bots > 0 else 0

    app.config["waiting_probabilities"] = {name:bot_waiting_probability for name in model_names}
    app.config["waiting_probabilities"][backend.Partner.Human] = bot_probability

    app.config["pairing_probabilities"] = {name:bot_probability for name in model_names}
    app.config["pairing_probabilities"][backend.Partner.Human] = bot_probability

    app.config["bots"] = models
    app.config["args"] = args

    init_database(params["db"]["location"], bot_probabilities=app.config["pairing_probabilities"])
    # logging.basicConfig(filename=params["logging"]["app_logs"], level=logging.INFO)
    socketio.run(app, host=args.host, port=args.port)
