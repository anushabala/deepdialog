__author__ = 'anushabala'
from argparse import ArgumentParser
import json
import os
from chat.modeling.tagger import Entity
from chat.modeling.chatbot import ChatBot
from transcript_utils import load_scenarios

SEQ_DELIM = "</s>"
SAY_DELIM = "SAY "
START = "START"
SELECT = "SELECT NAME"
COL_WIDTH="\"30%%\""
OUTPUT_STYLE="\"color:#4400FF\""
REGULAR_STYLE="\"color:#000000\""


def split_seq_into_sentences(seqs):
    seqs = seqs.split(SAY_DELIM)
    if seqs[0].strip() == START and len(seqs) > 1:
        seqs.pop(0)
    new_seqs = []
    for s in seqs:
        if SELECT in s:
            i = s.index(SELECT)
            new_seqs.append(s[:i])
            new_seqs.append(s[i:])
        else:
            new_seqs.append(s)
    return new_seqs


def replace_my_entities(s, bot):
    pass


def parse_transcripts(name, html_lines, replace_entities=False, scenario_ids=None, scenarios=None):
    results_json = json.load(open(name, 'r'))

    if replace_entities:
        assert len(scenario_ids) == len(results_json)

    ctr = 0
    for result in results_json:

        ctr += 1
        chat_html= ['<h3>Chat_%d</h3>' % ctr, '<table border=\"2\", style=\"border-collapse: collapse\">']
        model_input = [x.replace(SEQ_DELIM, "").strip() for x in result["x"]]
        model_out = result["y_pred"].split(SEQ_DELIM)
        model_ref = result["y_ref"].split(SEQ_DELIM)

        chat_html.append("<tr><td width=%s><b>Model Input</b></td>" % COL_WIDTH)
        chat_html.append("<td width=%s><b>Model Output</b></td>" % COL_WIDTH)
        chat_html.append("<td width=%s><b>Reference Output</b></td>" % COL_WIDTH)

        for i in range(0, len(model_input)):
            inp = split_seq_into_sentences(model_input[i])
            m_out = split_seq_into_sentences(model_out[i])
            if replace_entities:
                # do some stuff
                pass
            m_ref = split_seq_into_sentences(model_ref[i])

            chat_html.append("</tr><tr>")
            all_lines = [inp, m_out, m_ref]

            for (idx, lines) in enumerate(all_lines):
                style = REGULAR_STYLE
                if idx == 2:
                    style = OUTPUT_STYLE

                chat_html.append("<td width=%s style=%s> %s" % (COL_WIDTH, style, lines[0]))
                for line in lines[1:]:
                    chat_html.append("<br>%s" % line)
                chat_html.append("</td>")

        chat_html.append("</tr>")
        chat_html.append("</table>")
        html_lines.extend(chat_html)


def aggregate_chats(args):
    html = ['<!DOCTYPE html>','<html>']
    chats = []
    total = 0
    num_completed = 0
    if args.replace_entities:
        ids = [line.strip() for line in open(args.ids, 'r').readlines()]
        scenarios = load_scenarios(args.scenarios)
        parse_transcripts(args.results, chats, args.replace_entities, ids, scenarios)
    else:
        parse_transcripts(args.results, chats)

    html.extend(chats)
    html.append('</html>')
    return html

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--out', default='reports/', help='Directory to output report to')
    parser.add_argument('--filename', required=True, help='Filename to output report to')
    parser.add_argument('--results', required=True, help='File containing model results')
    parser.add_argument('--replace_entities', type=bool, default=False, help='Flag indicating whether to replace entity tags by entities or not')
    parser.add_argument('--ids', type=str, help='File to load scenario ids from')
    parser.add_argument('--scenarios', type=str, help='File to load scenarios from')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    outfile = open(os.path.join(args.out, args.filename), 'w')
    html_lines = aggregate_chats(args)

    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()

