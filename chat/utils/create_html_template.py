import json

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from transcript_utils import parse_transcript

NO_OUTCOME = "NO_OUTCOME"


def get_html_for_transcript(name, complete_html_lines, incomplete_html_lines, bot_type='human'):
    inp = open(name, 'r')
    chat_name = name[name.rfind('/')+1:]
    selections = {0:None, 1:None}
    chat_html= ['<h4>%s</h4>' % chat_name, '<table border=\"1\", style=\"border-collapse: collapse\">', '<tr>', '<td width=\"50%%\">']
    ended = 0
    closed_table = False
    transcript = parse_transcript(name, include_bots=True)
    if "BOT_TYPE" in transcript.keys():
        transcript_bot = transcript["BOT_TYPE"]
    else:
        transcript_bot = "human"

    all_users = list(set([x[0] for x in transcript["dialogue"]]))
    if len(transcript["dialogue"]) == 0 or len(all_users) != 2:
        return False, transcript_bot, transcript["scenario"], -1


    first_user = transcript["dialogue"][0][0]
    other_user_idx = all_users[0] if all_users[1] == first_user else all_users[1]
    if other_user_idx == 1 or other_user_idx == 0:
        other_user = "HUMAN"
    bot_idx = 1 - other_user_idx
    current_user = first_user
    chat_html.append("<b>%s</b></td><td width=\"50%%\"><b>%s</b></td></tr><tr><td width=\"50%%\">" % (first_user, other_user))

    for (user, line) in transcript["dialogue"]:
            if user != current_user:
                chat_html.append('</td>')
                if current_user != first_user:
                    chat_html.append('</tr><tr>')
                chat_html.append('<td width=\"50%%\">')
            elif current_user >= 0:
                chat_html.append('<br>')

            current_user = user
            chat_html.append(line)

    if current_user == first_user:
        chat_html.append('</td><td width=\"50%%\">LEFT</td></tr>')

    if not closed_table:
        chat_html.append('</table>')
    chat_html.append('<br>')
    completed = False if transcript["outcome"] == NO_OUTCOME else True
    if completed:
        chat_html.insert(0, '<div style=\"color:#0000FF\">')
    else:
        chat_html.insert(0, '<div style=\"color:#FF0000\">')
    chat_html.append('</div>')

    if bot_type == transcript_bot:
        if completed:
            complete_html_lines.extend(chat_html)
        else:
            incomplete_html_lines.extend(chat_html)

    return completed, transcript_bot, transcript["scenario"], bot_idx

def render_scenario(scenario, bot_idx, html_lines):
    chat_html = []
    bot_info = scenario["agents"][bot_idx]["info"]
    friends = scenario["agents"][bot_idx]["friends"]

    chat_html.append("<b>Bot information: </b>%s, %s, %s" % (bot_info["school"]["name"], bot_info["school"]["major"], bot_info["company"]["name"]))
    chat_html.append("<table><tr><td><b>Name</b></td><td><b>School</b></td><td><b>Major</b></td><td><b>Company</b></td></tr>")
    for friend in friends:
        chat_html.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (friend["name"], friend["school"]["name"]))
def aggregate_chats(dirname, bot_type="human", scenarios=None):
    html = ['<!DOCTYPE html>','<html>']
    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for f in os.listdir(args.dir):
        print f
        completed, partner_type = get_html_for_transcript(os.path.join(args.dir, f), completed_chats, incomplete_chats)
        if bot_type == partner_type:
            if completed:
                num_completed += 1
            total += 1
    html.extend(['<h3>Total number of chats: %d</h3>' % total, '<h3>Number of chats completed: %d</h3>' % num_completed])
    html.extend(completed_chats)
    html.extend(incomplete_chats)
    html.append('</html>')
    return html

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='transcripts', help='Path to directory containing transcripts')
    parser.add_argument('--output_dir', type=str, required=False, default='output', help='Path to directory to write HTML output to.')
    parser.add_argument('--output_name', type=str, required=True, help='Name of file to write report to')
    parser.add_argument('--scenarios', type=str, default='../friend_scenarios.json', help='JSON File containing scenarios')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    outfile = open(os.path.join(args.output_dir, args.output_name), 'w')
    scenarios = json.load(open(args.scenarios, 'r'))
    html_lines = aggregate_chats(args.dir, bot_type="LSTM_FEATURIZED", scenarios=scenarios)

    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()
