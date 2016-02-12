from flask import session, request
from flask import current_app as app
from flask.ext.socketio import emit, join_room, leave_room
from .. import socketio
from datetime import datetime
from .utils import get_backend
from .backend import Status
from .routes import userid

date_fmt = '%m-%d-%Y:%H-%M-%S'


@socketio.on('is_chat_valid')
def check_valid_chat():
    backend = get_backend()
    valid = backend.is_chat_valid(userid())
    msg = None
    if not valid:
        msg = backend.get_user_message(userid())
    return {'valid':valid, 'message':msg}


@socketio.on('check_status_change')
def check_status_change(data):
    backend = get_backend()
    current_status = Status.from_str(data['current_status'])

    if backend.is_status_unchanged(userid(), current_status):
        return {'status_change':False}
    else:
        return {'status_change':True}


@socketio.on('submit_task')
def submit_task(data):
    backend = get_backend()
    backend.submit_single_task(userid(), data)


@socketio.on('joined')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    start_chat()
    join_room(session["room"])
    emit_message_to_partner("Your partner has entered the room.", status_message=True)


@socketio.on('text')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    msg = message['msg']
    write_to_file(message['msg'])
    emit_message_to_self("You: {}".format(msg))
    emit_message_to_partner("Partner: {}".format(msg))


@socketio.on('pick')
def pick(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    backend = get_backend()
    chat_info = backend.get_chat_info(userid())
    restaurant_id = int(message['restaurant'])
    if restaurant_id == -1:
        return
    room = session["room"]
    restaurant, is_match = backend.pick_restaurant(userid(), restaurant_id)
    if is_match:
        emit_message_to_chat_room("Both users have selected restaurant: \"{}\"".format(restaurant), status_message=True)
        emit('endchat',
             {'message':"You've completed this task! Redirecting you..."},
             room=room)
    else:
        emit_message_to_partner("Your friend has selected restaurant: \"{}\"".format(restaurant), status_message=True)
        emit_message_to_self("You selected restaurant: \"{}\"".format(restaurant), status_message=True)
    write_outcome(restaurant, chat_info)


@socketio.on('left_room')
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session["room"]

    leave_room(room)
    # backend = get_backend()
    # backend.leave_room(userid())
    # backend.disconnect(userid())
    end_chat()


@socketio.on('disconnect')
def disconnect():
    """
    Called when user disconnects from any state
    :return: No return value
    """
    backend = get_backend()
    #todo check if state is chat here - actually just implement isvalidchat here
    backend.disconnect(userid())


def emit_message_to_self(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=request.sid)


def emit_message_to_chat_room(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=session["room"])


def emit_message_to_partner(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""    
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=session["room"],
         include_self=False)


def start_chat():
    chat_info = get_backend().get_chat_info(userid())

    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\tUser %s\tjoined\n" % (datetime.now().strftime(date_fmt),
                                            chat_info.scenario["uuid"],
                                            str(chat_info.agent_index)))
    outfile.close()


def end_chat():
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\n" % (datetime.now().strftime(date_fmt), app.config["user_params"]["logging"]["chat_delimiter"]))
    outfile.close()


def write_to_file(message):
    chat_info = get_backend().get_chat_info(userid())
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\tUser %s\t%s\n" %
                  (datetime.now().strftime(date_fmt), chat_info.scenario["uuid"],
                   str(chat_info.agent_index), message))
    outfile.close()


def write_outcome(name, chat_info):
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\tUser %s\tSelected restaurant:\t%s\n" %
                  (datetime.now().strftime(date_fmt), chat_info.scenario["uuid"], chat_info.agent_index, name))
