"""
USAGE

python cl_sort.py [source_data.txt] [target_data.txt] [vocab_for_sort.txt] [out_dir] [num_bins] [mix=True/False]
"""

import sys, os
from curriculum_learning import _bins_by_vocab_rank

def load_dataset(source, target):
	source_file = open(source)
	target_file = open(target)

	dataset = {}
	dataset["source"] = [line.split() for line in source_file.readlines()]
	dataset["target"] = [line.split() for line in target_file.readlines()]

	zipped = list(zip(*[dataset[k] for k in dataset.keys()]))

	return zipped

def draw_from_bins(bins):
    """
    reassemble complete corpus by iteratively (& uniformly) drawing from bins
    """

    #check for decreasing bin sizes
    valid = all(len(bins[i]) >= len(bins[i+1]) for i in range(len(bins)-1))
    if not valid:
        log("\nWith given thresholds, bin size constraints are not met")
        log("Data set has not been sorted.\nBin sizes:\n{}".format([len(el) for el in bins]))
        return None

    reassembled = []
    thresholds = calc_thresholds(bins)

    # create pools to draw from, each pool contains samples from all the 
    # allowed bins (up to its threshold)
    pools = []
    allowed_bins = [0]
    for t in thresholds:
        current_pool = []

        for b in allowed_bins:
            chunk = bins[b][:t]
            current_pool += chunk
            bins[b] = bins[b][t:]

        pools.append(current_pool)
        allowed_bins.append(allowed_bins[-1]+1)

    return pools

def calc_thresholds(bins):
    """
    a threshold determines the index for allowed bins from
    which are drawn from, after which the next bin is allowed
    """
    thresholds = []
    for i in range(len(bins)-1):
        t = len(bins[i]) - len(bins[i+1])
        thresholds.append(t)
    thresholds.append(len(bins[-1]))

    return thresholds


if __name__ == '__main__':
	source = sys.argv[1]
	target = sys.argv[2]
	sort_vocabulary = sys.argv[3]
	out_dir = sys.argv[4]
	num_bins = sys.argv[5]
	mix = sys.argv[6]
    normalize = sys.argv[7]

	dataset = load_dataset(source, target)
	bins = _bins_by_vocab_rank(dataset, sort_vocabulary, level="word", side="target", num_bins=int(num_bins), normalize=normalize)

	# each following bin has samples from the preceding ones 
	if mix == "True":
		bins = draw_from_bins(bins)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	print("Writing files to: {} ...".format(out_dir))

	for i in range(len(bins)):
		
		outfile_source = open("{}/src_{}.txt".format(out_dir, i), "w")
		outfile_target = open("{}/trg_{}.txt".format(out_dir, i), "w")

		for sent in bins[i]:
			outfile_source.write(" ".join(sent[0]))
			outfile_source.write("\n")

			outfile_target.write(" ".join(sent[1]))
			outfile_target.write("\n")

	print("... Done!")


