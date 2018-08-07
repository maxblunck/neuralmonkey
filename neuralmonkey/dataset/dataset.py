"""Implementation of the dataset class."""
import random
import collections
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from typeguard import check_argument_types
from neuralmonkey.curriculum_learning import create_bins_by_vocab_rank


class Dataset(collections.Sized):
    """Base Dataset class.

    This class serves as collection for data series for particular
    encoders and decoders in the model. If it is not provided a parent
    dataset, it also manages the vocabularies inferred from the data.

    A data series is either a list of strings or a numpy array.
    """

    def __init__(self,
                 name: str, series: Dict[str, List],
                 series_outputs: Dict[str, str],
                 preprocessors: List[Tuple[str, str, Callable]] = None
                ) -> None:
        """Create a dataset from the provided series of data.

        Arguments:
            name: The name for the dataset
            series: Dictionary from the series name to the actual data.
            series_outputs: Output files for target series.
            preprocessors: The definition of the preprocessors.
        """
        check_argument_types()

        self.name = name
        self._series = series
        self.series_outputs = series_outputs

        if preprocessors is not None:
            for src_id, tgt_id, function in preprocessors:
                if src_id == tgt_id:
                    raise Exception(
                        "Attempt to rewrite series '{}'".format(src_id))
                if src_id not in self._series:
                    raise Exception(
                        ("The source series ({}) of the '{}' preprocessor "
                         "is not defined in the dataset.").format(
                             src_id, str(function)))
                self._series[tgt_id] = [
                    function(item) for item in self._series[src_id]]

        self._check_series_lengths()

    def _check_series_lengths(self) -> None:
        """Check lenghts of series in the dataset.

        Raises:
            ValueError when the lengths in the dataset do not match.
        """
        lengths = [len(list(v)) for v in self._series.values()
                   if isinstance(v, (list, np.ndarray))]

        if len(set(lengths)) > 1:
            err_str = ["{}: {}".format(s, len(list(self._series[s])))
                       for s in self._series]
            raise ValueError("Lengths of data series must be equal. Were: {}"
                             .format(", ".join(err_str)))

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        if not list(self._series.values()):
            return 0

        first_series = next(iter(self._series.values()))
        return len(list(first_series))

    def has_series(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return name in self._series

    def maybe_get_series(self, name: str) -> Optional[Iterable]:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series or None if it does not exist.
        """
        return self._series.get(name, None)

    def get_series(self, name: str) -> Iterable:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series.

        Raises:
            KeyError if the series does not exists.
        """
        return self._series[name]

    @property
    def series_ids(self) -> Iterable[str]:
        return self._series.keys()

    def shuffle(self) -> None:
        """Shuffle the dataset randomly."""
        keys = list(self._series.keys())
        zipped = list(zip(*[self._series[k] for k in keys]))
        random.shuffle(zipped)
        for key, serie in zip(keys, list(zip(*zipped))):
            self._series[key] = serie

    def sort_by_curriculum(self) -> None:
        """curriculum learning"""
        keys = list(self._series.keys()) # ['source', 'target']
        zipped = list(zip(*[self._series[k] for k in keys])) # [([s1_fr],[s1_en]), ([s2_fr],[s2_en]), ...]

        #thresholds = [20, 40, 999, 10]
        thresholds = [3000, 5000, 7000, 12000, 99999] # rank_size better name for vocab_rank
        
        #criterion = "tar_len"
        #criterion = "vocab_rank"

        #bins = create_bins_by_sent_length(thresholds, zipped)
        bins, w2r = create_bins_by_vocab_rank(thresholds, zipped)

        for b in bins.keys():
            total_len=0
            for s in bins[b]:
                total_len += len(s[1])

            print("Average sent len of bin {} = {}".format(b, total_len/len(bins[b])))
        print("\n")  
          
        # determine sizes of bins and shuffle them
        sizes = [] #[48, 42, 10]
        for key in bins.keys():
            if bins[key] != None:
                sizes.append(len(bins[key]))
                random.shuffle(bins[key])
            else:
                sizes.append(0)

        # check for descending sizes of bins
        valid = all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))
        if not valid:
            print("\nWith given thresholds, the bin size requirements are not met. Bin sizes: {}".format(sizes))
            print("Corpus has not been sorted.\n")
            return

        print("Thresholds for curriculum sorted bins: {}".format(thresholds))
        print("Sizes of bins (sentences per bin): {}".format(sizes))

        # remove empty bins             
        for i in range(len(sizes)):
            if sizes[i] == 0:
                bins.pop(i)
                print("Deleted empty bin nr. {}".format(i))
    

        # sort data into batchs, from which the data can sequentially be drawn from
        current_limit_index = 0 
        allowed_bins = [0] # at start, only first bin allowed
        bin_index = 0
        num_bins = len(bins.keys())
        pools = []

        for i in range(num_bins):
            pool = [] # a pool contains sentences up to the next threshold from all allowed bins 

            for bin_key in allowed_bins:
                if i < num_bins - 1:
                    pool.append(bins[bin_key][sizes[current_limit_index+1] : sizes[current_limit_index]])
                else:
                    pool.append(bins[bin_key][0 : sizes[current_limit_index]])

            if bin_index < num_bins - 1:
                bin_index += 1
                allowed_bins.append(list(bins.keys())[bin_index])

            if current_limit_index < len(sizes):
                current_limit_index += 1

            flat_list = [sent for sents in pool for sent in sents]
            pools.append(flat_list)


        # sequentially draw from shuffled batchs
        sorted_data = []
        for pool in pools:
            random.shuffle(pool)
            for sent_pair in pool:
                sorted_data.append(sent_pair)


        #for s in sorted_data:
            #print([word2rank[word] for word in s[1]])
         #   print(len(s[1]))
            #print(get_max_rank(s[1], w2r))
            #print("\n")

        for key, serie in zip(keys, list(zip(*sorted_data))):
            self._series[key] = serie

    def batch_serie(self, serie_name: str,
                    batch_size: int) -> Iterable[Iterable]:
        """Split a data serie into batches.

        Arguments:
            serie_name: The name of the series
            batch_size: The size of a batch

        Returns:
            Generator yielding batches of the data from the serie.
        """
        buf = []
        for item in self.get_series(serie_name):
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def batch_dataset(self, batch_size: int) -> Iterable["Dataset"]:
        """Split the dataset into a list of batched datasets.

        Arguments:
            batch_size: The size of a batch.

        Returns:
            Generator yielding batched datasets.
        """
        keys = list(self._series.keys())
        batched_series = [self.batch_serie(key, batch_size) for key in keys]

        batch_index = 0
        for next_batches in zip(*batched_series):
            batch_dict = {key: data for key, data in zip(keys, next_batches)}
            dataset = Dataset(self.name + "-batch-{}".format(batch_index),
                              batch_dict, {})
            batch_index += 1
            yield dataset

    def add_series(self, name: str, series: List) -> None:
        if name in self._series:
            raise ValueError(
                "Can't add series that already exist: {}".format(name))
        self._series[name] = series

    def subset(self, start: int, length: int) -> "Dataset":
        subset_name = "{}.{}.{}".format(self.name, start, length)
        subset_outputs = {k: "{}.{:010}".format(v, start)
                          for k, v in self.series_outputs.items()}
        subset_series = {k: v[start:start + length]
                         for k, v in self._series.items()}

        return Dataset(subset_name, subset_series, subset_outputs)
