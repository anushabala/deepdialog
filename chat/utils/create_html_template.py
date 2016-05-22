__author__ = 'anushabala'

import os
from argparse import ArgumentParser


def parse_transcript(name, complete_html_lines, incomplete_html_lines):
    inp = open(name, 'r')
    chat_name = name[name.rfind('/')+1:]
    selections = {0:None, 1:None}
    chat_html= ['<h4>%s</h4>' % chat_name, '<table border=\"1\", style=\"border-collapse: collapse\">', '<tr>', '<td width=\"50%%\">']
    ended = 0
    current_user = -1
    first_user = -1
    closed_table = False
    for line in inp.readlines():
        line = line.strip().split('\t')
        print line
        if len(line) == 2:
            ended += 1
            # if ended == 2:
            #     chat_html.append('</table>')
            #     closed_table = True
        elif len(line) == 4 or len(line) == 5:
            if 'joined' not in line[-1]:
                user = int(line[2][-1])
                if first_user < 0:
                    first_user = user
                if user != current_user and current_user >= 0:
                    chat_html.append('</td>')
                    if current_user != first_user:
                        chat_html.append('</tr><tr>')
                    chat_html.append('<td width=\"50%%\">')
                elif current_user >= 0:
                    chat_html.append('<br>')

                current_user = user
                if len(line) == 4:
                    chat_html.append(line[-1])
                else:
                    selections[user] = line[-1]
                    chat_html.append('SELECT %s' % line[-1])

    if current_user == first_user:
        chat_html.append('</td><td width=\"50%%\">LEFT</td></tr>')
    completed = False
    if selections[0] == selections[1] and selections[0] is not None:
        completed = True

    if not closed_table:
        chat_html.append('</table>')
    chat_html.append('<br>')

    if completed:
        chat_html.insert(0, '<div style=\"color:#0000FF\">')
    else:
        chat_html.insert(0, '<div style=\"color:#FF0000\">')
    chat_html.append('</div>')

    if completed:
        complete_html_lines.extend(chat_html)
    else:
        incomplete_html_lines.extend(chat_html)

    return completed


def aggregate_chats(dirname):
    html = ['<!DOCTYPE html>','<html>']
    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for f in os.listdir(args.dir):
        print f
        completed = parse_transcript(os.path.join(args.dir, f), completed_chats, incomplete_chats)
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
    parser.add_argument('--output_file', type=str, required=True, help='Name of file to write report to')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    outfile = open(os.path.join(args.output_dir, args.output_file), 'w')
    html_lines = aggregate_chats(args.dir)

    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()
