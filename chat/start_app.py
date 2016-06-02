#!/bin/env python
import codecs
from collections import defaultdict
import sys
from app import create_app, socketio
import sqlite3
import os
import shutil
import json
from argparse import ArgumentParser
from chat.modeling.lexicon import Lexicon
from nn import spec as specutil
from nn.encoderdecoder import EncoderDecoderModel
from utils.dialogue_main import learn_bigram_model
from modeling.recurrent_box import BigramBox
from modeling.dialogue_tracker import add_arguments


# initialize database with table for chat rooms and active users
from modeling.tagger import EntityTagger


def init_database(db_file, bot_probability=0.0):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # number: room number, participants: number of participants (0 - 2)
    c.execute('''CREATE TABLE ActiveUsers (name text unique, status integer, status_timestamp integer, connected_status integer, connected_timestamp integer, message text, room_id integer, partner_id text, scenario_id text, agent_index integer, selected_index integer, single_task_id text, num_single_tasks_completed integer, num_chats_completed integer, cumulative_points integer, bonus integer)''')
    c.execute('''CREATE TABLE SingleTasks (name text, scenario_id text, selected_index integer, selected_restaurant text, start_text text)''')
    c.execute('''CREATE TABLE CompletedTasks (name text, mturk_code text, num_single_tasks_completed integer, num_chats_completed integer, bonus integer)''')
    c.execute('''CREATE TABLE Surveys (name text, partner_type text, how_mechanical integer, how_effective integer)''')
    c.execute('''CREATE TABLE ChatCounts (id integer, humans integer, bots integer, lstms_feat integer, lstms_unfeat integer, prob_bot real, prob_lstm_feat real, prob_lstm_unfeat real)''')
    c.execute('''INSERT INTO ChatCounts VALUES (1,0,0,0,0,?,?,?)''', (bot_probability["bot"], bot_probability["lstm_feat"], bot_probability["lstm_unfeat"]))
    #c.execute('''CREATE TABLE Chatrooms (room_id integer, scenario_id text)''')
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
    app.config["bots"] = defaultdict(None)
    app.config["lstms"] = defaultdict(None)
    app.config["bot_selections"] = defaultdict(None)
    app.config["lstm_selections"] = defaultdict(None)
    app.config["lexicon"] = Lexicon(scenarios_dict)

    num_bots = 0
    app.config["bot_waiting_probability"] = {}
    app.config["bot_probability"] = {}

    all_models = ["lstm_unfeat", "lstm_feat", "bot"]
    models = []
    if params["lstms"].get("use_feat_lstm"):
        num_bots += 1
        print "Loading model with features from from %s" % params["lstms"]["lstm_feat"]
        spec = specutil.load(params["lstms"]["lstm_feat"])
        # feat_model = EncoderDecoderModel(args, spec)
        cpt = learn_bigram_model(json.load(open(params["train"], 'r')))
        feat_model = BigramBox(cpt)
        app.config["lstm_feat"] = feat_model
        models.append("lstm_feat")
    else:
        app.config["lstm_feat"] = None
    if params["lstms"].get("use_unfeat_lstm"):
        num_bots += 1
        print "Loading model without features from %s" % params["lstms"]["lstm_unfeat"]
        spec = specutil.load(params["lstms"]["lstm_unfeat"])
        model = EncoderDecoderModel(args, spec)
        app.config["lstm_unfeat"] = model
        models.append("lstm_unfeat")
    else:
        app.config["lstm_unfeat"] = None

    if params.get("use_bots"):
        num_bots += 1
        app.config["use_bots"] = True
        models.append("bot")
    else:
        app.config["use_bots"] = False

    bot_waiting_probability = 1.0/num_bots if num_bots > 0 else 0
    bot_probability = 1.0/(num_bots+1) if num_bots > 0 else 0
    app.config["bot_waiting_probability"] = {m:bot_waiting_probability if m in models else 0 for m in all_models}
    app.config["bot_probability"] = {m:bot_probability if m in models else 0 for m in all_models}
    app.config["args"] = args
    init_database(params["db"]["location"], bot_probability=app.config["bot_probability"])
    # logging.basicConfig(filename=params["logging"]["app_logs"], level=logging.INFO)
    socketio.run(app, host=args.host, port=args.port)
