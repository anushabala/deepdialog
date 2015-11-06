__author__ = 'anushabala, mkayser'
'''
Provides functions to create training, test, and validation datasets given a file containing raw data from the drawing
task. Run python create_datasets.py -h for details.
'''
import argparse
import os
import re
import random
from treebank_tokenizer import TreebankWordTokenizer
from events import AbsoluteEventSequence, RelativeEventSequence, CursorEventSequence
from utils import read_csv

gif_pattern = r'(img_[0-9]+)\.gif'

def tokenize_description(tokenizer, text):
    def add_punct(line):
        if line.strip()[-1] not in ".!\"\',?":
            return line + " . "
        else:
            return line

    lines = text.split("</br>")
    tok_lines = [" ".join(tokenizer.tokenize(l)).lower() for l in lines]
    punct_tok_lines = [add_punct(l) for l in tok_lines if l.strip()]
    return " ".join(punct_tok_lines)

def write_data(header, data, images, image_field, commands_field, actions_field, output_file, mode="relative"):
    outfile = open(output_file, 'w')
    commands_index = header.index(commands_field)
    actions_index = header.index(actions_field)
    image_index = header.index(image_field)

    tokenizer = TreebankWordTokenizer()

    for row in data:
        image_url = row[image_index]
        image_key = re.search(gif_pattern, image_url).group(1)
        if image_key in images:
            commands = tokenize_description(tokenizer, row[commands_index])
            
            actions = row[actions_index]
            if mode=="relative":
                abs_seq = AbsoluteEventSequence.from_mturk_string(actions).canonicalize()
                rel_seq = RelativeEventSequence.from_absolute(abs_seq)
                actions = str(rel_seq)
            elif mode=="raw":
                actions = actions.replace("\r", "").replace("\n"," ")
            elif mode=="absolute":
                abs_seq = AbsoluteEventSequence.from_mturk_string(actions).canonicalize()
                actions = str(abs_seq)
            elif mode=="cursor":
                abs_seq = AbsoluteEventSequence.from_mturk_string(actions).canonicalize()
                rel_seq = CursorEventSequence.from_absolute(abs_seq)
                actions = str(rel_seq)
            else:
                raise Exception("Unexpected mode: {}".format(mode))
            outfile.write("%s\t%s\n" % (commands, actions))

    outfile.close()
    print "Created dataset at %s" % output_file


def split_images(header, data, image_field, train_ratio, test_ratio, val_ratio):
    images = set()

    image_index = header.index(image_field)
    for row in data:
        image_url = row[image_index]
        image_key = re.search(gif_pattern, image_url).group(1)
        images.add(image_key)

    images = sorted(list(images))
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    test_end = int(train_end + len(images) * test_ratio)
    val_end = len(images)

    train = images[0:train_end]
    test = images[train_end:test_end]
    val = images[test_end:val_end]

    return train, test, val

if __name__=="__main__":
    valid_modes = ["relative","cursor","raw","absolute"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="relative",
                        help="One of {}. If 'raw' is used, the sequence of commands from the "
                             "drawing task is used as is for model training/prediction. If 'relative' is used, the "
                             "commands are converted to relative commands. If cursor is used, commands are generated that."
                             "move a cursor to add the blocks. If 'absolute', the absolute coordinates are used" .format(valid_modes))
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file with Hamming distances for each example [or raw data from Turk drawing task; see the -compute_distance parameter")
    
    parser.add_argument("-rseed", type=int, default=0, help="Random seed")
    parser.add_argument("-train_ratio", type=float, default=0.67, help="Ratio (between 0 and 1) of examples to use for training")
    parser.add_argument("-test_ratio", type=float, default=0.17, help="Ratio (between 0 and 1) of examples to use for testing")
    parser.add_argument("-val_ratio", type=float, default=0.16, help="Ratio (between 0 and 1) of examples to use for validation")
    parser.add_argument("-output_dir", type=str, required=True, help="Output directory to write data to")
    parser.add_argument("-commands_field", type=str, default="Input.commands", help="Name of CSV field containing descriptions to arrange blocks")
    parser.add_argument("-image_field", type=str, default="Input.Image_url", help="Name of CSV field containing image URL")
    parser.add_argument("-draw_events_field", type=str, default="Answer.WritingTexts", help="Name of CSV field containing drawing task events")

    args = parser.parse_args()

    random.seed(args.rseed)

    data_header, all_data = read_csv(args.csv)
    train_images, test_images, val_images = split_images(data_header, all_data, args.image_field, args.train_ratio,
                                                         args.test_ratio, args.val_ratio)


    input_csv = args.csv
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = "%s/%s-%s-train.txt" % (output_dir, input_csv[input_csv.rfind('/'):].replace(".csv", ""), args.mode)
    test_file = "%s/%s-%s-test.txt" % (output_dir, input_csv[input_csv.rfind('/'):].replace(".csv", ""), args.mode)
    val_file = "%s/%s-%s-val.txt" % (output_dir, input_csv[input_csv.rfind('/'):].replace(".csv", ""), args.mode)

    write_data(data_header, all_data, train_images, args.image_field, args.commands_field, args.draw_events_field, train_file, args.mode)
    write_data(data_header, all_data, test_images, args.image_field, args.commands_field, args.draw_events_field, test_file, args.mode)
    write_data(data_header, all_data, val_images, args.image_field, args.commands_field, args.draw_events_field, val_file, args.mode)
