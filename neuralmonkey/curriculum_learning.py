"""
Curriculum learning
"""
import random

def create_bins_by_sent_length(thresholds, parallel_data):
    # create bins by target sentence length
    bins = dict().fromkeys(range(len(thresholds)))

    for sent_pair in parallel_data:
        for i in range(len(thresholds)): #for threshold in thresholds:
                if len(sent_pair[1]) <= thresholds[i]: # targent sent len <= threshold
                    if bins[i] != None:
                        bins[i].append(sent_pair)
                    else:
                        bins[i] = [sent_pair]
                    break
    return bins


def create_bins_by_vocab_rank(thresholds, parallel_data):
    # create bins by vocabulary rank
    ranks = dict.fromkeys(range(len(thresholds)))
    vocab_path = "../experiments/mle_curriculum/vocabs/target.vocab"

    # read vocab file and sort by ranks
    with open(vocab_path, encoding="utf-8") as vocab_file:
        lines = vocab_file.readlines()[1:]
        sorted_lines = sorted(lines, key=lambda x: int(x.split("\t")[1]))
        sorted_lines.reverse()

        print("Vocab size: {}".format(len(sorted_lines)))

        # if the size of last rank is larger than the vocab
        if thresholds[-1] > len(sorted_lines):
            thresholds[-1] = len(sorted_lines)

        for i in ranks.keys():
            ranks[i] = sorted_lines[:thresholds[i]]
            del sorted_lines[:thresholds[i]]


    # print stats
    for key in ranks.keys():
        print("Rank: {}".format(key))
        print("Size of rank: {}".format(len(ranks[key])))
        if ranks[key] != []:
            freq = [int(line.split("\t")[1]) for line in ranks[key]]
            sample = random.choice(ranks[key]).split("\t")
            print("Frequency range: {}:{}".format(max(freq), min(freq)))
            print("Random sample: {}, {}\n".format(sample[0], sample[1]))

    word2rank = dict((v.split("\t")[0],k) for k in ranks for v in ranks[k])

    bins = dict.fromkeys(ranks.keys())

    for sent_pair in parallel_data:
        try:
            sent_rank = get_max_rank(sent_pair[1], word2rank) #target sentence 
        except KeyError:
            sent_rank = list(ranks.keys())[-1] # if word is not in vocabulary, add to last rank

        if bins[sent_rank] != None:
            bins[sent_rank].append(sent_pair)
        else:
            bins[sent_rank] = [sent_pair]

    return bins, word2rank


def get_max_rank(sentence, word2rank):
    sent_rank = 0
    for word in sentence:
        word_rank = word2rank[word]
        if word_rank > sent_rank:
            sent_rank = word_rank
    return sent_rank