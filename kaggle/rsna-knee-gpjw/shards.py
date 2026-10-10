"""Read a cached series straight from the packed shard files, without unpacking them to local disk
(110 GB unpacked is more than a Kaggle session may write)."""
import json
import os

import numpy as np


class ShardLoader:
    """loader(series) -> uint8 volume; the `loader` argument of the reader's StudyDataset."""

    def __init__(self, index_json):
        self.index, self.opened = json.load(open(index_json)), {}

    def __call__(self, series):
        path = self.index[series]
        mine = self.opened.setdefault(os.getpid(), {})   # forked workers must not share a file handle
        z = mine.get(path)
        if z is None:
            z = mine[path] = np.load(path)
        return z[series]
