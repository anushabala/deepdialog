__author__ = 'mkayser'
from flask import g
from flask import current_app as app

from .backend import BackendConnection


def compute_agent_score(agent, restaurant):
    pr = restaurant["price_range"]
    c = restaurant["cuisine"]

    sf = agent["spending_func"]
    cf = agent["cuisine_func"]

    pr_points = next(obj["utility"] for obj in sf if obj["price_range"]==pr)
    c_points = next(obj["utility"] for obj in cf if obj["cuisine"]==c)

    return pr_points + c_points


def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        scenario_ids = sorted(app.config["scenarios"].keys())
        backend = g._backend = BackendConnection(app.config["user_params"], app.config["scenarios"],
                                                 app.config["paired_bots"], app.config["bot_selections"],
                                                 app.config["bots"], app.config["pairing_probabilities"],
                                                 app.config["waiting_probabilities"],app.config["lexicon"])
    return backend


def generate_outcome_key(user, partner, scenario_id):
    return (user, partner, scenario_id)


def generate_partner_key(user, partner, scenario_id):
    return (partner, user, scenario_id)
