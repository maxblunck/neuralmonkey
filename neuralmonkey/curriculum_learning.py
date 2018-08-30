#import pickle
import random
#import matplotlib 
#import matplotlib.pyplot as plt
from neuralmonkey.logging import log


def sort_data(data_series, vocabulary, criterion='sent_len', level='word', side='target', thresholds=None, num_bins=5):
    """
    criterion: sent_len, vocab_rank
    level: subword, word, ngram
    side: source, target
    """

    if level == "word":
        keys = ['source', 'target']
    elif level == "subword":
        keys = ['source', 'target', 'source_bpe', 'target_bpe']
    else:
        print("Error in level config")

    zipped = list(zip(*[data_series[k] for k in keys])) # [([s1_fr],[s1_en],[s1_fr_bpe],[s1_en_bpe]), ([s2_fr],[s2_en]), ...]

    if thresholds != None:
        thresholds = [int(t) for t in thresholds.split(",")]

    if criterion == 'sent_len':
        bins = _bins_by_sent_length(zipped, side, num_bins)
    elif criterion == 'vocab_rank':
        bins = _bins_by_vocab_rank(zipped, vocabulary, level, side, thresholds=thresholds, num_bins=num_bins)

    reassembled = _draw_from_bins(bins)

    # checks
    if reassembled == None or (len(reassembled) != len(zipped)):
        log("Unknown Error while sorting")
        return zipped, keys

    return reassembled, keys
  

def _draw_from_bins(bins):
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
    thresholds = _calc_thresholds(bins)

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

    # shuffle pools and flatten to full dataset
    for pool in pools:
        random.shuffle(pool)
        for sent_pair in pool:
            reassembled.append(sent_pair)

    return reassembled


def _calc_thresholds(bins):
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


def _auto_distribute_to_bins(parallel_corpus, num_bins):
    """
    distribute corpus to bins of descending sizes
    num_bins: number of bins 
    """
    uniform_bin_sizes = [int(len(parallel_corpus)/num_bins)] * num_bins

    # shift value: val by which first bin is multiplied,
    # gets lowered to 0.5 for last bin
    shift_val = 1.5 
    
    # calculate bin sizes
    bin_sizes = uniform_bin_sizes
    for index in range(len(uniform_bin_sizes)):
        bin_sizes[index] = int(uniform_bin_sizes[index] * shift_val)
        shift_val = shift_val - (1/(num_bins-1))

    # add the ones to 1. bin that were lost by cutting decimals
    bin_sizes[0] += len(parallel_corpus) - sum(bin_sizes)

    # add data to bins
    bins = []
    start_point = 0
    for end_point in bin_sizes:
        data_chunk = parallel_corpus[start_point:(start_point+end_point)]
        random.shuffle(data_chunk)
        bins.append(data_chunk)
        start_point = start_point + end_point

    return bins


def _bins_by_sent_length(parallel_corpus, side, num_bins):
    """
    LEVEL TO BE IMPLEMENTED
    returns parallel corpus sorted by either source or target side sentence length
    """
    if side == 'source':
        sorted_data = sorted(parallel_corpus, key=lambda x: int(len(x[0])))
    elif side == 'target':
        sorted_data = sorted(parallel_corpus, key=lambda x: int(len(x[1])))

    return _auto_distribute_to_bins(sorted_data, num_bins)


def _bins_by_vocab_rank(parallel_corpus, vocab_path, level, side, thresholds=None, num_bins=None, print_stats=True):
    """
    sorts dataset according to "min word freq per bin" in thresholds
    e.g: thresholds = [1000, 100, 10, 5, 0]
    rank1: only sents with words that appear at least 1000 times
    rank2: only sents with words that appear 1000-100 times 

    - sents with unknown words are added to last rank 
    - if no thresholds given, dataset is simply sorted by sents with high freq words first
    """

    vocab = _load_vocabulary(vocab_path)
    word_to_rank = _create_word_ranks(vocab, thresholds) # dict: k=word, v=rank

    if side == "source":

        if level == "word":
            sorted_dataset = sorted(parallel_corpus, key=lambda x: _get_sent_rank(x[0], word_to_rank, thresholds))
        elif level == "subword":
            sorted_dataset = sorted(parallel_corpus, key=lambda x: _get_sent_rank(x[2], word_to_rank, thresholds))

    elif side == "target":

        if level == "word":
            sorted_dataset = sorted(parallel_corpus, key=lambda x: _get_sent_rank(x[1], word_to_rank, thresholds))
        elif level == "subword":
            sorted_dataset = sorted(parallel_corpus, key=lambda x: _get_sent_rank(x[3], word_to_rank, thresholds))

    if thresholds != None:
        bins = _create_vocab_rank_bins(sorted_dataset, side, word_to_rank, thresholds)
    else:
        bins = _auto_distribute_to_bins(sorted_dataset, num_bins)

    # stats
    if print_stats == True:

        # auto thresholds
        auto_thresholds = []
        if thresholds == None:
            for bin in bins:
                min_b_freq = 999999
                for pair in bin:
                    freqs = [] #target!
                    if level == "word":
                        sent = pair[1]
                    elif level == "subword":
                        sent = pair[3]
                    for word in sent:
                        try:
                            freqs.append(vocab[word])
                        except KeyError:
                            freqs.append(1)
                    min_s_freq = min(freqs)
                    if min_s_freq < min_b_freq:
                        min_b_freq = min_s_freq
                auto_thresholds.append(min_b_freq)

        log("Total size of dataset: {}".format(len(sorted_dataset)))
        log("\nHighest word frequency: {}".format(max(vocab.values())))     
        log("\nGiven thresholds: {}".format(thresholds))
        log("\nAuto thresholds: {}".format(auto_thresholds)) 
        log("\nBin sizes: {}".format([len(bin) for bin in bins]))
        for i in range(len(bins)):
            if level == "word":
                log("\nBin {}:\n size: {}\n example: {}\n".format(i, len(bins[i]), " ".join(random.choice(bins[i])[1])))
            elif level == "subword":
                log("\nBin {}:\n size: {}\n example: {}\n".format(i, len(bins[i]), " ".join(random.choice(bins[i])[3])))

    return bins


def _create_vocab_rank_bins(sorted_dataset, side, word_to_rank, thresholds):
    bins = dict()
    for pair in sorted_dataset:

        if side == 'source':

            if level == "word":
                rank = _get_sent_rank(pair[0], word_to_rank, thresholds)
            elif level == "subword":
                rank = _get_sent_rank(pair[2], word_to_rank, thresholds)

        elif side == 'target':

            if level == "word":
                rank = _get_sent_rank(pair[1], word_to_rank, thresholds)
            elif level == "subword":
                rank = _get_sent_rank(pair[3], word_to_rank, thresholds)
        try:
            bins[rank].append(pair)
        except KeyError:
            bins[rank] = [pair]

    return list(bins.values())


def _get_sent_rank(sentence, word_to_rank, thresholds=None):
    max_rank = 0
    for word in sentence:
        try:
            word_rank = word_to_rank[word]
        except KeyError:
            if thresholds != None:
                return len(thresholds)-1
            else:
                return len(word_to_rank.values()) 
        if word_rank > max_rank:
            max_rank = word_rank
    return max_rank


def _create_word_ranks(vocab, thresholds=None):
    """
    smooth or discrete
    """
    ordered_vocab = sorted(vocab, key=vocab.get, reverse=True) # by desc freq
    desc_frequencies = sorted(vocab.values(), reverse=True)
    ranks = dict()

    if thresholds != None:
        for i in range(len(ordered_vocab)):
            for t in thresholds:
                if desc_frequencies[i] > t:
                    ranks[ordered_vocab[i]] = thresholds.index(t)
                    break
    else:
        for word in ordered_vocab:
            #print("{}: {}".format(word, vocab[word]))
            ranks[word] = ordered_vocab.index(word)

    return ranks


def _load_vocabulary(path):
    """
    reads vocabulary file as saved by NM
    returns dict with k=word, v=freq
    """
    vocab_dict = dict()
    with open(path, encoding="utf-8") as vocab_file:
        lines = vocab_file.readlines()[1:]
        for line in lines:
            splitted = line.split("\t")
            vocab_dict[splitted[0]] = int(splitted[1].strip("\n")) # key: word, val: freq
    return vocab_dict


def _plot_stats(data, criterion):
    if criterion == 'sent_len':
        fig, ax = plt.subplots()
        ax.plot(range(len(data)), [len(pair[1]) for pair in data])
        ax.set(xlabel='data sample', ylabel='target length')
        ax.grid()
        fig.savefig("data_distribution.png")
        plt.show()


# if __name__ == '__main__':
#     corpus_file = open("zipped_corpus.pickle", 'rb')
#     corpus = pickle.load(corpus_file) # [([s1_fr],[s1_en]), ([s2_fr],[s2_en]), ...]

#     #vocabulary = "source.vocab"
#     vocabulary = "target.vocab"
#     sort_data(corpus, vocabulary, criterion="vocab_rank", level="word", side="target", num_bins=7)#, thresholds=[78, 19, 5, 1, 0])

