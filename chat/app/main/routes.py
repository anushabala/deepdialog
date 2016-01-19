from flask import g, session, redirect, url_for, render_template, request
from flask import current_app as app
from . import main
from .forms import LoginForm, RestaurantForm
import sqlite3
import random
import time
from .utils import get_backend

ctr = 0


# todo try and use one connection everywhere, put code to find unpaired users into single function
@main.route('/', methods=['GET', 'POST'])
def index():
    """"Login form to enter a room."""
    form = LoginForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        add_new_user(session["name"])
        room, scenario_id = find_room_if_possible(session["name"])
        if room:
            return redirect(url_for('.chat'))
        else:
            return redirect(url_for('.waiting'))
    elif request.method == 'GET':
        form.name.data = session.get('name', '')
    return render_template('index.html', form=form)


@main.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat room. The user's name and room must be stored in
    the session."""
    name = session.get('name', None)
    room = session.get('room', None)
    agent_number = session.get('agent_number')
    scenario_id = session.get('scenario_id', None)
    form=RestaurantForm()
    if form.validate_on_submit():
        app.logger.debug("Testing logger: POST request, successfully validated.")
        return redirect(url_for('.index'))
    elif request.method == 'GET':
        app.logger.debug("Testing logger: chat requested.")
        if name is None or room is None or scenario_id is None:
            return redirect(url_for('.index'))
        else:
            scenario = app.config["scenarios"][scenario_id]
            return render_template('chat.html', name=name, room=room, scenario=scenario, agent_number=agent_number, form=form)
    else:
        app.logger.debug("Testing logger: POST request but not validated.")
        


@main.route('/single_task')
# todo: something like this needs to happen when a single task is submitted, too
def waiting():
    name = session.get('name', None)
    global ctr
    while ctr < app.config["user_params"]["WAITING_TIME"]:
        time.sleep(1)
        ctr += 1
        room, scenario_id = find_room_if_possible(name)
        if room:
            ctr = 0
            return redirect(url_for('.chat'))
        else:
            return redirect(url_for('.waiting'))
    ctr = 0
    return render_template('single_task.html')


def add_new_user(username):
    backend = get_backend()
    backend.create_user_if_necessary(username)


def find_room_if_possible(username):
    backend = get_backend()
    room, scenario_id, agent_number = backend.find_room_for_user_if_possible(username)
    app.logger.debug("User %s has agent ID %d" % (session.get('name'), agent_number))
    if room:
        session["room"] = room
        session["scenario_id"] = scenario_id
        session["agent_number"] = agent_number
    return (room, scenario_id)
