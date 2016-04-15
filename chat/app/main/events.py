from flask import session, request
from flask import current_app as app
from flask.ext.socketio import emit, join_room, leave_room
from .. import socketio
from datetime import datetime
from .utils import get_backend
from .backend import Status
from .routes import userid
import logging

date_fmt = '%m-%d-%Y:%H-%M-%S'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def userid_prefix():
    return userid()[:6]


@socketio.on('connect', namespace='/main')
def connect():
    backend = get_backend()
    backend.connect(userid())
    logger.info("User %s established connection on non-chat template" % userid_prefix())


@socketio.on('connect', namespace='/chat')
def connect():
    backend = get_backend()
    backend.connect(userid())
    logger.info("User %s established connection on chat template" % userid_prefix())


@socketio.on('is_chat_valid', namespace='/chat')
def check_valid_chat(data):
    backend = get_backend()

    if backend.is_chat_valid(userid()):
        logger.debug("Chat is still valid for user %s" % userid_prefix())
        return {'valid': True}
    else:
        logger.info("Chat is not valid for user %s" % userid_prefix())
        return {'valid': False, 'message': backend.get_user_message(userid())}


@socketio.on('check_status_change', namespace='/main')
def check_status_change(data):
    backend = get_backend()
    assumed_status = Status.from_str(data['current_status'])

    if backend.is_status_unchanged(userid(), assumed_status):
        logger.debug("User %s status unchanged. Status: %s" % (userid_prefix(), Status._names[assumed_status]))
        return {'status_change': False}
    else:
        logger.info("User %s status changed from %s" % (userid_prefix(), Status._names[assumed_status]))
        return {'status_change': True}


@socketio.on('submit_task', namespace='/main')
def submit_task(data):
    backend = get_backend()
    logger.debug("User %s submitted single task. Form data: %s" % (userid_prefix(), str(data)))
    backend.submit_single_task(userid(), data)


@socketio.on('joined', namespace='/chat')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    start_chat()
    join_room(session["room"])
    logger.debug("User %s joined chat room %d" % (userid_prefix(), session["room"]))
    emit_message_to_partner("Your friend has entered the room.", status_message=True)


@socketio.on('text', namespace='/chat')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    msg = message['msg']
    write_to_file(msg)
    logger.debug("User %s said: %s" % (userid_prefix(), msg))
    emit_message_to_self("You: {}".format(msg))
    emit_message_to_partner("Friend: {}".format(msg))


@socketio.on('pick', namespace='/chat')
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
    logger.debug("User %s in room %d selected: %s" % (userid_prefix(), room, restaurant))
    if is_match:
        logger.info("User %s selection matches with partner selection" % userid_prefix())
        emit_message_to_chat_room("Both users have selected: \"{}\"".format(restaurant), status_message=True)
        emit('endchat',
             {'message': "You've completed this task! Redirecting you..."},
             room=room)
    else:
        logger.debug("User %s selection doesn't match with partner selection" % userid_prefix())
        emit_message_to_partner("Your friend has selected: \"{}\"".format(restaurant), status_message=True)
        emit_message_to_self("You selected: \"{}\"".format(restaurant), status_message=True)
    write_outcome(restaurant_id, restaurant, chat_info)


@socketio.on('disconnect', namespace='/chat')
def disconnect():
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session["room"]

    leave_room(room)
    backend = get_backend()
    # backend.leave_room(userid())
    backend.disconnect(userid())
    logger.info("User %s disconnected from chat and left room %d" % (userid_prefix(), room))
    end_chat()


@socketio.on('disconnect', namespace='/main')
def disconnect():
    """
    Called when user disconnects from any state
    :return: No return value
    """
    backend = get_backend()
    backend.disconnect(userid())
    logger.info("User %s disconnected" % (userid_prefix()))


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
    outfile.write("%s\t%s\tUser %s has user ID %s\n" % (datetime.now().strftime(date_fmt),
                                                      chat_info.scenario["uuid"],
                                                      str(chat_info.agent_index),
                                                      userid()))
    outfile.close()


def end_chat():
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write(
        "%s\t%s\n" % (datetime.now().strftime(date_fmt), app.config["user_params"]["logging"]["chat_delimiter"]))
    outfile.close()


def write_to_file(message):
    chat_info = get_backend().get_chat_info(userid())
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\tUser %s\t%s\n" %
                  (datetime.now().strftime(date_fmt), chat_info.scenario["uuid"],
                   str(chat_info.agent_index), message))
    outfile.close()


def write_outcome(idx, name, chat_info):
    outfile = open('%s/ChatRoom_%s' % (app.config["user_params"]["logging"]["chat_dir"], str(session["room"])), 'a+')
    outfile.write("%s\t%s\tUser %s\tSelected %d:\t%s\n" %
                  (datetime.now().strftime(date_fmt), chat_info.scenario["uuid"], chat_info.agent_index, idx, name))
