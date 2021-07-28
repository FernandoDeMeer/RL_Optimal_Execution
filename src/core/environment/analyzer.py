

class DataAnalyzer:

    def __init__(self, algo_ids):

        self.algo_ids = algo_ids

        self.data = {}

        for algo_id in algo_ids:
            self.data[algo_id] = {"pxs" : []}
            self.data[algo_id] = {"qty" : []}

    def record_step(self, algo_id, key_name, value):
        self.data[algo_id][key_name].append(value)

    def calc(self):
        pass
