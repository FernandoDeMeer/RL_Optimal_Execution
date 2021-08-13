import numpy as np


class TradesMonitor:

    def __init__(self, algo_ids):

        self.algo_ids = algo_ids
        self.reset()

    def record_step(self, algo_id, key_name, value):
        self.data[algo_id][key_name].append(value)

    def calc_vwaps(self):
        vwaps = dict()
        for algo_id in self.algo_ids:
            prices = np.array(self.data[algo_id]['pxs'])
            volumes = np.array(self.data[algo_id]['qty'])
            vwaps[algo_id] = np.sum(prices * volumes) / np.sum(volumes)
        return vwaps

    def calc_IS(self):
        IS = dict()
        for algo_id in self.algo_ids:
            prices = np.array(self.data[algo_id]['pxs'])
            volumes = np.array(self.data[algo_id]['qty'])
            arrival_prices = np.array(self.data[algo_id]['arrival'])
            IS[algo_id] = (arrival_prices - prices)*volumes
        return IS

    def calc(self):
        pass

    def reset(self):
        self.data = {}
        for algo_id in self.algo_ids:
            self.data[algo_id] = {"pxs": [], "qty": [], "t": [], "arrival": []}
