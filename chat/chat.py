#!/bin/env python
from collections import defaultdict
from app import create_app, socketio
import sqlite3
import os
import shutil
import json
from argparse import ArgumentParser
import logging


# initialize database with table for chat rooms and active users
from app.main.tagger import EntityTagger


def init_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # number: room number, participants: number of participants (0 - 2)
    c.execute('''CREATE TABLE ActiveUsers (name text unique, status integer, status_timestamp integer, connected_status integer, connected_timestamp integer, message text, room_id integer, partner_id text, scenario_id text, agent_index integer, selected_index integer, single_task_id text, num_single_tasks_completed integer, num_chats_completed integer, cumulative_points integer, bonus integer)''')
    c.execute('''CREATE TABLE SingleTasks (name text, scenario_id text, selected_index integer, selected_restaurant text, start_text text)''')
    c.execute('''CREATE TABLE CompletedTasks (name text, mturk_code text, num_single_tasks_completed integer, num_chats_completed integer, bonus integer)''')
    c.execute('''CREATE TABLE Surveys (name text, partner_type text, how_mechanical integer, how_effective integer)''')
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
    parser.add_argument('--log', help="File to log app output to", type=str, default="chat.log")
    args = parser.parse_args()
    params_file = args.p
    with open(params_file) as fin:
        params = json.load(fin)

    if os.path.exists(params["db"]["location"]):
        os.remove(params["db"]["location"])

    init_database(params["db"]["location"])
    clear_data(params["logging"]["chat_dir"])
    templates_dir = params["templates_dir"]
    app = create_app(debug=True, templates_dir=templates_dir)

    with open(params["scenarios_json_file"]) as fin:
        scenarios = json.load(fin)
        scenarios_dict = {v["uuid"]:v for v in scenarios}

    app.config["user_params"] = params
    app.config["scenarios"] = scenarios_dict
    app.config["outcomes"] = defaultdict(lambda : -1)
    app.config["bots"] = defaultdict(None)
    app.config["bot_selections"] = defaultdict(None)
    app.config["tagger"] = EntityTagger(scenarios_dict, params["bots"]["templates"])

    # logging.basicConfig(filename=params["logging"]["app_logs"], level=logging.INFO)

    socketio.run(app, host=args.host)
