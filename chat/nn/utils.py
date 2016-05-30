__author__ = 'anushabala'

def sentence_pairs_to_indices(in_vocabulary, out_vocabulary, pairs, eos_on_output):
  all_x_inds = []
  all_y_inds = []
  #print pairs
  for x,y in pairs:
    x0_inds = in_vocabulary.sentence_to_indices(x)
    y0_inds = [-1 for i in x0_inds]
    y1_inds = out_vocabulary.sentence_to_indices(y, add_eos=eos_on_output)
    x1_inds = [-1 for i in y1_inds]

    all_x_inds.extend(x0_inds + x1_inds)
    all_y_inds.extend(y0_inds + y1_inds)
  return (all_x_inds, all_y_inds)


def sentence_pairs_to_indices_for_eval(in_vocabulary, out_vocabulary, pairs, eos_on_output):
  results = []
  for x,y in pairs:
    x_inds = in_vocabulary.sentence_to_indices(x)
    y_inds = out_vocabulary.sentence_to_indices(y, add_eos=eos_on_output)
    results.append((x_inds,y_inds))

  return results