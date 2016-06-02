from __future__ import with_statement
import sqlite3
import datetime
import time
import logging
import uuid
import random

from flask import Markup

from .backend_utils import UserChatSession, SingleTaskSession, WaitingSession, FinishedSession
from chat.modeling.chatbot import ChatBot
from chat.modeling.lstmbot import ModelBot

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


class Status(object):
    Waiting, Chat, SingleTask, Finished, Survey = range(5)
    _names = ["waiting", "chat", "single_task", "finished", "survey"]

    @staticmethod
    def from_str(s):
        if Status._names.index(s) == 0:
            return Status.Waiting
        if Status._names.index(s) == 1:
            return Status.Chat
        if Status._names.index(s) == 2:
            return Status.SingleTask
        if Status._names.index(s) == 3:
            return Status.Finished
        if Status._names.index(s) == 4:
            return Status.Survey

    def to_str(self):
        return self._names[self]

class Partner(object):
    BaselineBot = 'bot'
    Human = 'human'
    LSTMBot = 'lstm'

def current_timestamp_in_seconds():
    return int(time.mktime(datetime.datetime.now().timetuple()))


class BadChatException(Exception):
    pass


class NoSuchUserException(Exception):
    pass


class UnexpectedStatusException(Exception):
    def __init__(self, found_status, expected_status):
        self.expected_status = expected_status
        self.found_status = found_status


class ConnectionTimeoutException(Exception):
    pass


class InvalidStatusException(Exception):
    pass


class StatusTimeoutException(Exception):
    pass


class User(object):
    def __init__(self, row):
        self.name = row[0]
        self.status = row[1]
        self.status_timestamp = row[2]
        self.connected_status = row[3]
        self.connected_timestamp = row[4]
        self.message = row[5]
        self.room_id = row[6]
        self.partner_id = row[7]
        self.scenario_id = row[8]
        self.agent_index = row[9]
        self.selected_index = row[10]
        self.single_task_id = row[11]
        self.num_single_tasks_completed = row[12]
        self.num_chats_completed = row[13]
        self.cumulative_points = row[14]
        self.bonus = row[15]


class Messages(object):
    ChatExpired = "Darn, you ran out of time! Waiting for a new chat..."
    PartnerConnectionTimeout = "Your friend's connection has timed out! Waiting for a new chat..."
    ConnectionTimeout = "Your connection has timed out! Waiting for a new chat..."
    YouLeftRoom = "You have left the room. Waiting for a new chat..."
    PartnerLeftRoom = "Your friend has left the room! Waiting for a new chat..."


class BackendConnection(object):
    def __init__(self, config, scenarios, bots, bot_selections, featurized_lstm_model, unfeaturized_lstm,
                 bot_waiting_probabilities, lexicon, args=None):
        self.config = config
        self.featurized_lstm = featurized_lstm_model
        self.unfeaturized_lstm = unfeaturized_lstm
        self.conn = sqlite3.connect(config["db"]["location"])
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT prob_bot, prob_lstm_feat, prob_lstm_unfeat FROM ChatCounts WHERE id=1''')
            probs = cursor.fetchone()
            self.bot_probability = probs[0]
            self.lstm_probability = probs[1]
            self.lstm_unfeaturized_probability = probs[2]
            self.human_probability = 1 - (self.bot_probability + self.lstm_probability + self.lstm_unfeaturized_probability)
        self.args = args
        self.lexicon = lexicon

        self.do_survey = True if "end_survey" in config.keys() and config["end_survey"] == 1 else False
        self.scenarios = scenarios
        self.bots = bots
        self.bot_selections = bot_selections
        self.waiting_bot_probability = bot_waiting_probabilities["bot"]
        self.waiting_lstm_probability = bot_waiting_probabilities["lstm_feat"]
        self.waiting_lstm_unfeaturized_probability = bot_waiting_probabilities["lstm_unfeat"]

    def close(self):
        self.conn.close()
        self.conn = None

    def create_user_if_necessary(self, username):
        with self.conn:
            cursor = self.conn.cursor()
            now = current_timestamp_in_seconds()
            logger.debug("Created user %s" % username[:6])
            cursor.execute('''INSERT OR IGNORE INTO ActiveUsers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (username, Status.Waiting, now, 0, now, "", -1, "", "", -1, -1, "", 0, 0, 0, 0))

    def is_status_unchanged(self, userid, assumed_status):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    logger.debug("Checking whether status has changed from %s for user %s" % (
                        Status._names[assumed_status], userid[:6]))
                    u = self._get_user_info(cursor, userid, assumed_status=assumed_status)
                    if u.status == Status.Waiting:
                        logger.debug("User %s is waiting. Checking if other users are available for chat..")
                        self.attempt_join_room(userid)
                        u = self._get_user_info(cursor, userid, assumed_status=assumed_status)
                    logger.debug("Returning TRUE (user status hasn't changed)")
                    return True
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    logger.warn(
                        "Caught %s while getting status for user %s. Returning FALSE (user status has changed)" % (
                            type(e).__name__, userid[:6]))
                    if isinstance(e, UnexpectedStatusException):
                        logger.warn("Found status %s, expected (assumed) status %s" % (
                            Status._names[e.found_status], Status._names[e.expected_status]))
                    return False

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def is_connected(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid)
                    return True if u.connected_status == 1 else False
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    return False

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_updated_status(self, userid):
        try:
            logger.debug("Getting current status for user %s" % userid[:6])
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid, assumed_status=None)
                    logger.debug("Got user info for user %s without exceptions. Returning status %s" % (
                        userid[:6], Status._names[u.status]))
                    return u.status
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    logger.warn("Caught %s while getting status for user %s" % (type(e).__name__, userid[:6]))
                    if isinstance(e, UnexpectedStatusException):
                        logger.warn(
                            "Unexpected behavior: got UnexpectedStatusException while getting user status")  # this should never happen
                    # Handle timeouts by performing the relevant update
                    u = self._get_user_info_unchecked(cursor, userid)
                    logger.debug("Unchecked user status for user %s: %s" % (userid[:6], u.status))
                    if u.status == Status.Waiting:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info(
                                "User %s had connection timeout in waiting state. Updating connection status to connected to reenter waiting state." % userid[
                                                                                                                                                       :6])
                            self._update_user(cursor, userid, connected_status=1, status=Status.Waiting,
                                              num_single_tasks_completed=0)
                            return u.status
                        logger.info("User %s had status timeout in waiting state." % userid[:6])
                        self._transition_to_single_task(cursor, userid)
                        return Status.SingleTask
                    elif u.status == Status.SingleTask:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info(
                                "User %s had connection timeout in single task state. Updating connection status to connected and reentering waiting state." % userid[
                                                                                                                                                               :6])
                            self._update_user(cursor, userid, connected_status=1, status=Status.Waiting,
                                              num_single_tasks_completed=0)
                            return Status.Waiting
                        return u.status  # this should never happen because single tasks can't time out
                    elif u.status == Status.Chat:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info(
                                "User %s had connection timeout in chat state. Updating connection status to connected and reentering waiting state." % userid[
                                                                                                                                                        :6])
                            message = Messages.PartnerConnectionTimeout
                            self._update_user(cursor, userid, num_single_tasks_completed=0)
                        else:
                            logger.info(
                                "Chat timed out for user %s. Leaving chat room and entering waiting state.." % userid[
                                                                                                               :6])
                            message = Messages.ChatExpired

                        if self.is_user_partner_bot(userid):
                            user_bot = self.bots[userid]
                            user_bot.end_chat()
                            self.bots[userid] = None
                            self.bot_selections[userid] = None

                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=message,
                                                                 partner_message=message)
                        return Status.Waiting
                    elif u.status == Status.Finished:
                        logger.info(
                            "User %s was previously in finished state. Updating to waiting state with connection status = connected." % userid[
                                                                                                                                        :6])
                        self._update_user(cursor, userid, connected_status=1, status=Status.Waiting, message='',
                                          num_single_tasks_completed=0)
                        if self.is_user_partner_bot(userid):
                            user_bot = self.bots[userid]
                            user_bot.end_chat()
                            self.bots[userid] = None
                            self.bot_selections[userid] = None

                        return Status.Waiting
                    elif u.status == Status.Survey:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info('ConnectionTimeOutException for user %s in survey state. Updating to connected and reentering waiting state.' % userid[:6])
                            self._update_user(cursor, userid, connected_status=1, status=Status.Waiting,
                                              num_single_tasks_completed=0)

                            return Status.Waiting
                        return Status.Survey # this should never happen because surveys can't time out
                    else:
                        raise Exception("Unknown status: {} for user: {}".format(u.status, userid))

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def _transition_to_single_task(self, cursor, userid):
        logger.info("Updating status for user %s from Waiting to SINGLE TASK" % userid[:6])
        scenario_id = random.choice(list(self.scenarios.keys()))
        logger.debug("Chose scenario %s for user" % scenario_id)
        my_agent_index = random.choice([0, 1])
        self._update_user(cursor, userid,
                          status=Status.SingleTask,
                          single_task_id=scenario_id,
                          agent_index=my_agent_index,
                          selected_index=-1,
                          message="")

    def _end_chat_and_transition_to_waiting(self, cursor, userid, partner_id, message, partner_message):
        logger.info("Removing users %s and %s from chat room - transition to WAIT" % (userid[:6], partner_id[:6]))
        self._update_user(cursor, userid,
                          status=Status.Waiting,
                          room_id=-1,
                          connected_status=1,
                          message=message)
        if partner_id is not None:
            self._update_user(cursor, partner_id,
                              status=Status.Waiting,
                              room_id=-1,
                              message=partner_message)

    def get_chat_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting chat info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                num_seconds_remaining = (self.config["status_params"]["chat"][
                                             "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                scenario = self.scenarios[u.scenario_id]
                return UserChatSession(u.room_id, u.agent_index, scenario, num_seconds_remaining)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_single_task_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting single task info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.SingleTask)
                num_seconds_remaining = (self.config["status_params"]["single_task"][
                                             "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                scenario = self.scenarios[u.single_task_id]
                return SingleTaskSession(scenario, u.agent_index, num_seconds_remaining)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_waiting_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting waiting session info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Waiting)
                num_seconds = (self.config["status_params"]["waiting"][
                                   "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                return WaitingSession(u.message, num_seconds)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_finished_info(self, userid, from_mturk=False):
        def _generate_mturk_code(user_info):
            return "MTURK_TASK_{}".format(str(uuid.uuid4().hex))

        def _add_finished_task_row(cursor, userid, mturk_code, num_single_tasks_completed, num_chats_completed,
                                   has_bonus):
            logger.info(
                "Adding row into CompletedTasks: userid={},mturk_code={},numsingle={},numchats={},grant_bonus={}".format(
                    userid[:6],
                    mturk_code,
                    num_single_tasks_completed,
                    num_chats_completed,
                    has_bonus))
            cursor.execute('INSERT INTO CompletedTasks VALUES (?,?,?,?,?)',
                           (userid, mturk_code, num_single_tasks_completed, num_chats_completed, 1 if has_bonus else 0))

        try:
            logger.info("Trying to get finished session info for user %s" % userid[:6])
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Finished)
                num_seconds = (self.config["status_params"]["finished"][
                                   "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                if from_mturk:
                    logger.info("Generating mechanical turk code for user %s" % userid[:6])
                    mturk_code = _generate_mturk_code(u)
                    message = u.message
                    logger.info("User %s bonus status: %s" % (userid[:6], str(u.bonus)))
                    # if u.bonus == 1:
                    #     logger.info("User %s has bonus" % userid[:6])
                        # message += "<p><b>Good job on this negotiation! You'll receive a bonus on Mechanical Turk.</b></p>"
                    _add_finished_task_row(cursor, userid, mturk_code, u.num_single_tasks_completed,
                                           u.num_chats_completed, u.bonus)
                    self._update_user(cursor, userid, bonus=0)
                    return FinishedSession(Markup(message), num_seconds, mturk_code)
                self._update_user(cursor, userid, bonus=0)
                return FinishedSession(Markup(u.message), num_seconds)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def is_chat_valid(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                except UnexpectedStatusException:
                    return False
                except StatusTimeoutException:
                    u = self._get_user_info_unchecked(cursor, userid)
                    logger.debug("User {} had status timeout.".format(u.name[:6]))
                    self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=Messages.YouLeftRoom,
                                                             partner_message=Messages.PartnerLeftRoom)
                    return False
                except ConnectionTimeoutException:
                    return False

                if not self.is_user_partner_bot(userid):
                    try:
                        u2 = self._get_user_info(cursor, u.partner_id, assumed_status=Status.Chat)
                    except UnexpectedStatusException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, None, message=Messages.PartnerLeftRoom,
                                                                 partner_message=None)
                        return False
                    except StatusTimeoutException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=Messages.ChatExpired,
                                                                 partner_message=Messages.ChatExpired)
                        return False
                    except ConnectionTimeoutException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id,
                                                                 message=Messages.PartnerLeftRoom,
                                                                 partner_message=Messages.YouLeftRoom)
                        return False

                    return u.room_id == u2.room_id

                return True

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return False

    def connect(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid,
                                  connected_status=1)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def disconnect(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid,
                                  connected_status=0)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def _ensure_not_none(self, v, exception_class):
        if v is None:
            logger.warn("None: ", v)
            logger.warn("Raising exception %s" % type(exception_class).__name__)
            raise exception_class()
        else:
            return v

    def pick_restaurant(self, userid, restaurant_index):
        def _get_points(scenario, agent_index, restaurant_name):
            result = scenario["connection"]["info"]["name"]
            if result == restaurant_name:
                return 5
            return 0

        def _user_finished(cursor, userid, prev_points, my_points, other_points, prev_chats_completed, optimal_choice=None):
            message = "Great, you've finished the chat! You scored {} points and your friend scored {} points.".format(
                my_points, other_points)
            logger.info("Updating user %s to status FINISHED from status chat, with total points %d+%d=%d" %
                        (userid[:6], prev_points, my_points, prev_points + my_points))
            new_status = Status.Survey if self.do_survey else Status.Finished
            if optimal_choice:
                # message += "<p>The best restaurant you could have chosen, given your preferences, was <b>%s</b> (%d points).</p>" % (optimal_choice["name"], optimal_choice["points"])
                self._update_user(cursor, userid, status=new_status, message=message,
                                  cumulative_points=prev_points + my_points,
                                  num_chats_completed=prev_chats_completed + 1,
                                  bonus=0)
            else:
                # message += "<p>You chose the best restaurant given your preferences!<p>"
                self._update_user(cursor, userid, status=new_status, message=message,
                                  cumulative_points=prev_points + my_points,
                                  num_chats_completed=prev_chats_completed + 1,
                                  bonus=1
                                  )

        def _is_optimal_choice(scenario, agent_index, restaurant_name, user_points):
            # top_choice = scenario["agents"][agent_index]["sorted_restaurants"][0]["name"]
            # best_points = scenario["agents"][agent_index]["sorted_restaurants"][0]["utility"]
            # second_choice = scenario["agents"][agent_index]["sorted_restaurants"][1]["name"]
            # optimal = {"name":top_choice, "points":best_points}
            result = scenario["connection"]["info"]["name"]
            return restaurant_name == result

        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid, selected_index=restaurant_index)
                u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                P = u.cumulative_points
                scenario = self.scenarios[u.scenario_id]
                restaurant_name = scenario["agents"][u.agent_index]["friends"][restaurant_index]["name"]
                if self.is_user_partner_bot(userid):
                    if userid not in self.bot_selections.keys():
                        return restaurant_name, False
                    other_name = self.bot_selections[userid]
                    if restaurant_name == other_name:
                        Pdelta = _get_points(scenario, u.agent_index, restaurant_name)
                        is_optimal = _is_optimal_choice(scenario, u.agent_index, restaurant_name, Pdelta)
                        if is_optimal:
                            _user_finished(cursor, userid, P, Pdelta, Pdelta, u.num_chats_completed)
                            return restaurant_name, True
                    else:
                        return restaurant_name, False

                cursor.execute(
                    "SELECT name, selected_index,cumulative_points,num_chats_completed FROM ActiveUsers WHERE room_id=? AND name!=?",
                    (u.room_id, userid))
                other_userid, other_restaurant_index, other_P, other_num_chats_completed = self._ensure_not_none(
                    cursor.fetchone(), BadChatException)
                if other_restaurant_index == -1:
                    return restaurant_name, False
                other_name = scenario["agents"][1-u.agent_index]["friends"][other_restaurant_index]["name"]
                if restaurant_name == other_name:
                    # Match
                    logger.info("User %s restaurant selection matches with partner's. Selected restaurant: %s" % (
                        userid[:6], restaurant_name))
                    Pdelta = _get_points(scenario, u.agent_index, restaurant_name)
                    other_Pdelta = _get_points(scenario, 1 - u.agent_index, restaurant_name)
                    logger.info("User %s got %d points, User %s got %d points" % (
                        userid[:6], Pdelta, other_userid[:6], other_Pdelta))

                    is_optimal = _is_optimal_choice(scenario, u.agent_index, restaurant_name, Pdelta)
                    other_is_optimal = _is_optimal_choice(scenario, 1 - u.agent_index,
                                                                                restaurant_name, other_Pdelta)
                    if is_optimal:
                        _user_finished(cursor, userid, P, Pdelta, other_Pdelta, u.num_chats_completed)
                    else:
                        _user_finished(cursor, userid, P, Pdelta, other_Pdelta, u.num_chats_completed)
                    if other_is_optimal:
                        _user_finished(cursor, other_userid, other_P, other_Pdelta, Pdelta, other_num_chats_completed)
                    else:
                        _user_finished(cursor, other_userid, other_P, other_Pdelta, Pdelta, other_num_chats_completed)
                    return restaurant_name, True
                else:
                    logger.debug("User %s selection (%d) doesn't match with partner's selection (%d). " %
                                 (userid[:6], restaurant_index, other_restaurant_index))
                    # Non match
                    return restaurant_name, False

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def _validate_status_or_throw(self, assumed_status, status):
        logger.debug("Validating status: User status {}, assumed status {}".format(status, assumed_status))
        if status != assumed_status:
            logger.warn(
                "Validating status: User status {}, assumed status {} Raising UnexpectedStatusException".format(status,
                                                                                                                assumed_status))
            raise UnexpectedStatusException(status, assumed_status)
        return

    def _assert_no_status_timeout(self, status, status_timestamp):
        N = self.config["status_params"][Status._names[status]]["num_seconds"]
        if N < 0:  # don't timeout for some statuses
            logger.debug("Checking for status timeout: no status timeout for status {}".format(Status._names[status]))
            return
        num_seconds_remaining = (N + status_timestamp) - current_timestamp_in_seconds()

        if num_seconds_remaining >= 0:
            logger.debug("No status timeout")
            logger.debug(
                "Checking for timeout of status '%s': Seconds for status: %d Status timestamp: %d Seconds remaining: %d" % (
                    Status._names[status], N, status_timestamp, num_seconds_remaining))
            return
        else:
            logger.info(
                "Checking for timeout of status '%s': Seconds for status: %d Status timestamp: %d Seconds remaining: %d" % (
                    Status._names[status], N, status_timestamp, num_seconds_remaining))
            logger.warn("Checking for status timeout: Raising StatusTimeoutException")
            raise StatusTimeoutException()

    def _assert_no_connection_timeout(self, connection_status, connection_timestamp):
        logger.debug("Checking for connection timeout: Connection status %d" % connection_status)
        if connection_status == 1:
            logger.debug("No connection timeout")
            return
        else:
            N = self.config["connection_timeout_num_seconds"]
            num_seconds_remaining = (N + connection_timestamp) - current_timestamp_in_seconds()
            if num_seconds_remaining >= 0:
                logger.debug("Timeout limit: %d Status timestamp: %d Seconds remaining: %d" % (
                    N, connection_timestamp, num_seconds_remaining))
                logger.debug("No connection timeout")
                return
            else:
                logger.info("Timeout limit: %d Status timestamp: %d Seconds remaining: %d" % (
                    N, connection_timestamp, num_seconds_remaining))
                logger.warn("Checking for connection timeout: Raising ConnectionTimeoutException")
                raise ConnectionTimeoutException()

    def _update_user(self, cursor, userid, **kwargs):
        if "status" in kwargs:
            logger.info("Updating status for user %s to %s" % (userid[:6], Status._names[kwargs["status"]]))
            kwargs["status_timestamp"] = current_timestamp_in_seconds()
        if "connected_status" in kwargs:
            logger.info("Updating connected status for user %s to %d" % (userid[:6], kwargs["connected_status"]))
            kwargs["connected_timestamp"] = current_timestamp_in_seconds()
        keys = sorted(kwargs.keys())
        values = [kwargs[k] for k in keys]
        set_string = ", ".join(["{}=?".format(k) for k in keys])

        cursor.execute("UPDATE ActiveUsers SET {} WHERE name=?".format(set_string), tuple(values + [userid]))

    def _get_user_info_unchecked(self, cursor, userid):
        cursor.execute("SELECT * FROM ActiveUsers WHERE name=?", (userid,))
        x = cursor.fetchone()
        u = User(self._ensure_not_none(x, NoSuchUserException))
        return u

    def _get_user_info(self, cursor, userid, assumed_status=None):
        logger.debug("Getting info for user {} (assumed status: {})".format(userid[:6], assumed_status))
        u = self._get_user_info_unchecked(cursor, userid)
        if assumed_status is not None:
            self._validate_status_or_throw(assumed_status, u.status)
        self._assert_no_connection_timeout(u.connected_status, u.connected_timestamp)
        self._assert_no_status_timeout(u.status, u.status_timestamp)
        return u

    def get_user_message(self, userid):
        with self.conn:
            cursor = self.conn.cursor()
            u = self._get_user_info_unchecked(cursor, userid)
            return u.message

    def submit_survey(self, userid, data):
        def _user_finished(userid):
            # message = "<h3>Great, you've finished this task!</h3>"
            logger.info("Updating user %s to status FINISHED from status survey" % userid)
            self._update_user(cursor, userid, status=Status.Finished)
            if self.is_user_partner_bot(userid):
                self.bots[userid] = None
                self.bot_selections[userid] = None

        try:
            with self.conn:
                cursor = self.conn.cursor()
                partner_name = Partner.Human
                if self.is_user_partner_bot(userid):
                    partner_name = self.get_user_bot(userid).name
                cursor.execute('INSERT INTO Surveys VALUES (?,?,?,?)',
                               (userid, partner_name, data['question1'], data['question2']))
                _user_finished(userid)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def submit_single_task(self, userid, user_input):
        def _complete_task_and_wait(cursor, userid, num_finished):
            message = "Great, you've finished {} exercises! Waiting a few seconds for someone to chat with...".format(
                num_finished)
            logger.info("Updating user info for user %s after single task completion - transition to WAIT" % userid[:6])
            self._update_user(cursor, userid, status=Status.Waiting, message=message,
                              num_single_tasks_completed=num_finished)

        def _complete_task_and_finished(cursor, userid, num_finished):
            message = "Great, you've finished {} exercises!".format(num_finished)
            logger.info(
                "Updating user info for user %s after single task completion - transition to FINISHED" % userid[:6])
            self._update_user(cursor, userid, status=Status.Finished, message=message,
                              num_single_tasks_completed=num_finished)

        def _log_user_submission(cursor, userid, scenario_id, user_input):
            logger.debug("Logging submission from user %s to database. Submission: %s" % (userid[:6], str(user_input)))
            cursor.execute('INSERT INTO SingleTasks VALUES (?,?,?,?,?)',
                           (userid, scenario_id, user_input["restaurant_index"], user_input["restaurant"],
                            user_input["starter_text"]))

        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.SingleTask)
                _log_user_submission(cursor, userid, u.single_task_id, user_input)
                if u.num_single_tasks_completed == self.config["status_params"]["single_task"]["max_tasks"] - 1:
                    _complete_task_and_finished(cursor, userid, u.num_single_tasks_completed + 1)
                else:
                    _complete_task_and_wait(cursor, userid, u.num_single_tasks_completed + 1)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def leave_room(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                logger.info("Removing user %s and partner from chat" % userid[:6])
                u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                message = Messages.YouLeftRoom
                partner_message = Messages.PartnerLeftRoom
                logger.debug("Successfully retrieved user and partner information")
                self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=message,
                                                         partner_message=partner_message)
        except UnexpectedStatusException:
            logger.warn(
                "leave_room(): Got UnexpectedStatusException while trying to get user information for user %s" % userid[
                                                                                                                 :6])
            return
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def attempt_join_room(self, userid, use_bot=True):
        def _redistribute_probabilities(probs_and_ratios):
            lower_bound = min(1/len(probs_and_ratios) - 0.1, 0.05)
            upper_bound = min(1/len(probs_and_ratios) + 0.1, 0.95)
        def _change_bot_probability(cursor):
            humans, bots, lstms, total = _get_num_chats(cursor)
            if total == 0:  # only change probabilities every 50 chats or so
                return
            #todo fix this later (lstms)
            # humans = float(humans)/total
            # bots = float(bots)/total
            # lstms = float(lstms)/total
            #
            # if 0.2 <= bots <= 0.4 and 0.2 <= humans <= 0.4 and 0.2 <= lstms <= 0.4:
            #     return
            # if bots <= 0.2:
            #     if humans >= 0.4 and self.human_probability >= 0.15:
            #         self.human_probability -= 0.1
            #     if lstms >= 0.4 >= 0.15:
            #         self.lstm_probability -= 0.1
            #     self.bot_probability = 1 - (self.lstm_probability+ self.human_probability)
            # if humans <= 0.2:
            #     if bots >= 0.4 and self.bot_probability >=0.15:
            #         self.bot_probability -= 0.1
            #     if lstms >= 0.4 and self.lstm_probability >= 0.4
            # humans = float(humans)/total
            # if total % 50 == 0:
            #     if 0.25 <= humans <= 0.4:
            #         return
            #     if humans < 0.4 and self.bot_probability >= 0.15:
            #         self.bot_probability -= 0.1
            #         #print "Decreased bot probability to ", self.bot_probability
            #         cursor.execute('''UPDATE ChatCounts SET prob=? WHERE id=?''', (self.bot_probability, 1))
            #     if humans > 0.6 and self.bot_probability <= 0.85:
            #         self.bot_probability += 0.1
            #         #print "Increased bot probability to ", self.bot_probability
            #         cursor.execute('''UPDATE ChatCounts SET prob=? WHERE id=?''', (self.bot_probability, 1))

        def _get_num_chats(cursor):
            cursor.execute("SELECT humans, bots, lstms_feat, lstms_unfeat FROM ChatCounts WHERE id=?", (1,))
            x = cursor.fetchone()
            humans = x[0]
            bots = x[1]
            lstms_feat = x[2]
            lstms_unfeat = x[3]
            return humans, bots, lstms_feat, lstms_unfeat, humans+bots

        def _add_bot_chat(cursor):
            cursor.execute("SELECT bots from ChatCounts WHERE id=?", (1,))
            x = cursor.fetchone()
            bots = x[0]
            cursor.execute("UPDATE ChatCounts SET bots=? WHERE id=?", (bots+1, 1))

        def _add_human_chat(cursor):
            cursor.execute("SELECT humans from ChatCounts WHERE id=?", (1,))
            x = cursor.fetchone()
            humans = x[0]
            cursor.execute("UPDATE ChatCounts SET humans=? WHERE id=?", (humans+1, 1))

        def _add_lstm_chat(cursor, featurized=True):
            key = "lstms_feat" if featurized else "lstms_unfeat"
            cursor.execute("SELECT %s from ChatCounts WHERE id=?" % key, (1,))
            x = cursor.fetchone()
            lstms = x[0]
            cursor.execute("UPDATE ChatCounts SET %s=? WHERE id=?" % key, (lstms+1, 1))

        def _pair_with_bot(userid):
            scenario_id = random.choice(list(self.scenarios.keys()))
            next_room_id = _get_max_room_id(cursor) + 1
            my_agent_index = random.choice([0, 1])
            # todo fix this!
            bot = ChatBot(self.scenarios[scenario_id], 1 - my_agent_index, self.tagger)
            self.bots[userid] = bot
            self._update_user(cursor, userid,
                              status=Status.Chat,
                              room_id=next_room_id,
                              partner_id=0,
                              scenario_id=scenario_id,
                              agent_index=my_agent_index,
                              selected_index=-1,
                              message="")
            return next_room_id

        def _pair_with_lstm(userid, model=self.featurized_lstm):
            scenario_id = random.choice(list(self.scenarios.keys()))
            next_room_id = _get_max_room_id(cursor) + 1
            my_agent_index = random.choice([0,1])
            name = "LSTM_FEATURIZED" if model == self.featurized_lstm else "LSTM_UNFEATURIZED"
            bot = ModelBot(self.scenarios[scenario_id], 1-my_agent_index, model, self.lexicon, name=name, args=self.args)
            self.bots[userid] = bot
            self._update_user(cursor, userid,
                              status=Status.Chat,
                              room_id=next_room_id,
                              partner_id=0,
                              scenario_id=scenario_id,
                              agent_index=my_agent_index,
                              selected_index=-1,
                              message="")
            return next_room_id

        def _get_other_waiting_users(cursor, userid):
            cursor.execute("SELECT name FROM ActiveUsers WHERE name!=? AND status=? AND connected_status=1",
                           (userid, Status.Waiting))
            userids = [r[0] for r in cursor.fetchall()]
            return userids

        def _get_max_room_id(cursor):
            cursor.execute("SELECT MAX(room_id) FROM ActiveUsers", ())
            return cursor.fetchone()[0]

        try:
            logger.debug("Attempting to find room for user %s" % userid[:6])
            with self.conn:
                cursor = self.conn.cursor()

                # _change_bot_probability(cursor)

                u = self._get_user_info(cursor, userid, assumed_status=Status.Waiting)
                others = _get_other_waiting_users(cursor, userid)
                logger.debug("Found %d other unpaired users" % len(others))
                if len(others) == 0:
                    r = random.random()
                    if use_bot:
                        if r < self.waiting_bot_probability:
                            room_id = _pair_with_bot(userid)
                            _add_bot_chat(cursor)
                            print "Pairing with baseline", r
                            return room_id
                        elif self.waiting_bot_probability <= r < self.waiting_bot_probability + self.waiting_lstm_probability:
                            room_id = _pair_with_lstm(userid, self.featurized_lstm)
                            print "Pairing with featurized LSTM", r
                            _add_lstm_chat(cursor)
                            return room_id
                        elif self.waiting_bot_probability + self.waiting_lstm_probability <= r < self.waiting_bot_probability + self.waiting_lstm_probability + self.waiting_lstm_unfeaturized_probability:
                            room_id = _pair_with_lstm(userid, self.unfeaturized_lstm)
                            print "Pairing with unfeaturized LSTM", r
                            _add_lstm_chat(cursor, featurized=False)
                            return room_id
                        else:
                            return None
                    else:
                        return None
                if len(others) > 0:
                    r = random.random()
                    if use_bot:
                        if r < self.bot_probability:
                            room_id = _pair_with_bot(userid)
                            _add_bot_chat(cursor)
                            return room_id
                        elif self.bot_probability <= r < self.bot_probability + self.lstm_probability:
                            room_id = _pair_with_lstm(userid, self.featurized_lstm)
                            _add_lstm_chat(cursor)
                            return room_id
                        elif self.bot_probability + self.lstm_probability <= r < self.bot_probability + self.lstm_probability + self.lstm_unfeaturized_probability:
                            room_id = _pair_with_lstm(userid, self.unfeaturized_lstm)
                            _add_lstm_chat(cursor, featurized=False)
                            return room_id

                    _add_human_chat(cursor)
                    scenario_id = random.choice(list(self.scenarios.keys()))
                    other_userid = random.choice(others)
                    next_room_id = _get_max_room_id(cursor) + 1
                    logger.info("Paired user %s with user %s, room %d, scenario %s" % (
                        userid[:6], other_userid[:6], next_room_id, scenario_id))
                    logger.debug("Updating users with new chat information")
                    my_agent_index = random.choice([0, 1])
                    self._update_user(cursor, other_userid,
                                      status=Status.Chat,
                                      room_id=next_room_id,
                                      partner_id=userid,
                                      scenario_id=scenario_id,
                                      agent_index=1 - my_agent_index,
                                      selected_index=-1,
                                      message="")
                    self._update_user(cursor, userid,
                                      status=Status.Chat,
                                      room_id=next_room_id,
                                      partner_id=other_userid,
                                      scenario_id=scenario_id,
                                      agent_index=my_agent_index,
                                      selected_index=-1,
                                      message="")
                    return next_room_id
                else:
                    return None
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def is_user_partner_bot(self, userid):
        return userid in self.bots.keys() and self.bots[userid] is not None

    def get_user_bot(self, userid):
        if userid not in self.bots.keys():
            return None
        return self.bots[userid]

    def make_bot_selection(self, userid, bot_selection):
        def _get_points(scenario, agent_index, restaurant_name):
            result = scenario["connection"]["info"]["name"]
            if result == restaurant_name:
                return 5
            return 0

        def _user_finished(cursor, userid, prev_points, my_points, other_points, prev_chats_completed, optimal_choice=None):
            message = "Great, you've finished the chat! You scored {} points and your friend scored {} points.".format(
                my_points, other_points)
            logger.info("Updating user %s to status FINISHED from status chat, with total points %d+%d=%d" %
                        (userid[:6], prev_points, my_points, prev_points + my_points))
            new_status = Status.Survey if self.do_survey else Status.Finished
            if optimal_choice:
                # message += "<p>The best restaurant you could have chosen, given your preferences, was <b>%s</b> (%d points).</p>" % (optimal_choice["name"], optimal_choice["points"])
                self._update_user(cursor, userid, status=new_status, message=message,
                                  cumulative_points=prev_points + my_points,
                                  num_chats_completed=prev_chats_completed + 1,
                                  bonus=0)
            else:
                # message += "<p>You chose the best restaurant given your preferences!<p>"
                self._update_user(cursor, userid, status=new_status, message=message,
                                  cumulative_points=prev_points + my_points,
                                  num_chats_completed=prev_chats_completed + 1,
                                  bonus=1
                                  )

        def _is_optimal_choice(scenario, agent_index, restaurant_name, user_points):
            # top_choice = scenario["agents"][agent_index]["sorted_restaurants"][0]["name"]
            # best_points = scenario["agents"][agent_index]["sorted_restaurants"][0]["utility"]
            # second_choice = scenario["agents"][agent_index]["sorted_restaurants"][1]["name"]
            # optimal = {"name":top_choice, "points":best_points}
            result = scenario["connection"]["info"]["name"]
            return restaurant_name == result

        try:
            with self.conn:
                self.bot_selections[userid] = bot_selection
                cursor = self.conn.cursor()
                # self._update_user(cursor, userid, selected_index=restaurant_index)
                u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                P = u.cumulative_points
                scenario = self.scenarios[u.scenario_id]
                if u.selected_index == -1:
                    return bot_selection, False
                user_selection = scenario["agents"][u.agent_index]["friends"][u.selected_index]["name"]
                if user_selection == bot_selection:
                    Pdelta = _get_points(scenario, u.agent_index, user_selection)
                    is_optimal = _is_optimal_choice(scenario, u.agent_index, user_selection, Pdelta)
                    if is_optimal:
                        _user_finished(cursor, userid, P, Pdelta, Pdelta, u.num_chats_completed)
                        return bot_selection, True
                else:
                    return bot_selection, False

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")



