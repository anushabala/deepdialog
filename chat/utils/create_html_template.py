__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from transcript_utils import parse_transcript_dating, parse_transcript, load_scenarios, NO_OUTCOME


def create_html_lines(chat_name, transcript, complete_html, incomplete_html):
    selections = {0:None, 1:None}
    chat_html= ['<h4>%s</h4>' % chat_name, '<table border=\"1\", style=\"border-collapse: collapse\">', '<tr>', '<td width=\"50%%\">']
    ended = 0
    current_user = -1
    print transcript["dialogue"]
    if len(transcript["dialogue"]) == 0:
        print "Empty transcript, %s" % chat_name
        return False
    first_user = transcript["dialogue"][0][0]

    print transcript["dialogue"]
    for (user, line) in transcript["dialogue"]:
        print line
        if user != current_user and current_user >= 0:
            chat_html.append('</td>')
            if current_user != first_user:
                chat_html.append('</tr><tr>')
            chat_html.append('<td width=\"50%%\">')
        elif current_user >= 0:
            chat_html.append('<br>')

        chat_html.append(line)
        current_user = user

    if current_user == first_user:
        chat_html.append('</td><td width=\"50%%\">  </td></tr>')
    else:
        chat_html.append('</tr>')

    chat_html.append('</table>')
    chat_html.append('<br>')
    completed = True if transcript['outcome'] != NO_OUTCOME else False

    if completed:
        chat_html.insert(0, '<div style=\"color:#0000FF\">')
    else:
        chat_html.insert(0, '<div style=\"color:#FF0000\">')
    chat_html.append('</div>')

    if completed:
        complete_html.extend(chat_html)
    else:
        incomplete_html.extend(chat_html)

    return completed


def aggregate_chats(scenario_type='friends'):
    html = ['<!DOCTYPE html>','<html>']
    complete_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for f in os.listdir(args.dir):
        print f
        if scenario_type == 'friends':
            transcript = parse_transcript(os.path.join(args.dir, f))
        else:
            transcript = parse_transcript_dating(os.path.join(args.dir, f), scenarios)
        completed = create_html_lines(f, transcript, complete_chats, incomplete_chats)
        if completed:
            num_completed += 1
        total += 1
    html.extend(['<h3>Total number of chats: %d</h3>' % total, '<h3>Number of chats completed: %d</h3>' % num_completed])
    html.extend(complete_chats)
    html.extend(incomplete_chats)
    html.append('</html>')
    return html

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='transcripts', help='Path to directory containing transcripts')
    parser.add_argument('--output_dir', type=str, required=False, default='output', help='Path to directory to write HTML output to.')
    parser.add_argument('--output_name', type=str, required=True, help='Name of file to write report to')
    parser.add_argument('--scenarios_file', type=str, default='../friend_scenarios.json', help='Path to file containing scenarios')
    parser.add_argument("--scenario_type", type=str, default='friends', choices=['friends', 'dating'], help='Type of scenario (friends or dating)')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    scenarios = load_scenarios(args.scenarios_file)
    outfile = open(os.path.join(args.output_dir, args.output_name+'.html'), 'w')
    html_lines = aggregate_chats(args.scenario_type)

    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()
