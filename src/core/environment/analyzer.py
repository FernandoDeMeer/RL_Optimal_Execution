import numpy as np


class DataAnalyzer:

    def __init__(self, algo_ids):

        self.algo_ids = algo_ids
        self.reset()

    def record_step(self, algo_id, key_name, value):
        self.data[algo_id][key_name].append(value)

    def calc_vwaps(self):
        vwaps = dict()
        for algo_id in self.algo_ids:
            vwap = np.sum((self.data[algo_id]['pxs'] * self.data[algo_id]['qty']) /
                          np.sum(self.data[algo_id]['qty']))
            vwaps[algo_id] = vwap
        return vwaps

    def calc(self):
        pass

    def reset(self):

        self.data = {}
        for algo_id in self.algo_ids:
            self.data[algo_id] = {"pxs" : []}
            self.data[algo_id] = {"qty" : []}
            self.data[algo_id] = {"t" : []}
