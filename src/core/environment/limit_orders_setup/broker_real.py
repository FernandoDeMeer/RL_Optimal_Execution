import copy
from abc import ABC
from datetime import datetime
from decimal import Decimal


def calc_volume_weighted_price_from_trades(trades):
    """ Calculates volume weighted prices from a trade """

    volume = 0
    price = 0
    for idx in range(len(trades)):
        volume = volume + trades[idx]['quantity']
        price = price + (trades[idx]['price'] * trades[idx]['quantity'])
    price = price / volume
    return float(price), float(volume)


def place_order(lob, dt, order):
    """ places an order into the LOB """
    ord = order.copy()
    trade_message = None
    if ord['quantity'] > 0:
        if ord['type'] == 'limit':
            lob_temp = copy.deepcopy(lob)
            trades, _ = lob_temp.process_order(ord, False, False)
        else:
            lob_temp = copy.deepcopy(lob)
            trades, _ = lob_temp.process_order(ord, True, False)
        if trades:
            vol_wgt_price, vol = calc_volume_weighted_price_from_trades(trades)
            msg = 'trade'
        else:
            vol_wgt_price, vol, msg = 0, 0, 'no_trade'
        trade_message = {'timestamp': datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f'),
                         'message': msg,
                         'type': order['type'],
                         'price': vol_wgt_price,
                         'quantity': Decimal(str(vol)),
                         'target_quantity': order['quantity']}
    return trade_message


class Broker(ABC):
    """ Currently only for placing trades and getting volume weighted execution prices """

    def __init__(self, data_feed,):

        self.data_feed = data_feed
        self.benchmark_algo = None
        self.rl_algo = None
        self.hist_dict = {'timestamp': [],
                          'benchmark_lob': [],
                          'rl_lob': []}
        self.remaining_order = {'benchmark_algo': [],
                                'rl_algo': []}
        self.trade_logs = {'benchmark_algo': [],
                           'rl_algo': []}

    def reset(self, algo,):
        """ Resetting the Broker class """

        # reset the Broker logs
        if type(algo).__name__ != 'RLAlgo':
            self.hist_dict['timestamp'] = []
            self.hist_dict['benchmark_lob'] = []
            self.remaining_order['benchmark_algo'] = []
            self.trade_logs['benchmark_algo']= []
        else:
            self.hist_dict['timestamp'] = []
            self.hist_dict['rl_lob'] = []
            self.remaining_order['rl_algo'] = []
            self.trade_logs['rl_algo'] = []

        # update to the first instance of the datafeed & record this
        self.data_feed.reset(time=algo.start_time)
        dt, lob = self.data_feed.next_lob_snapshot()

        algo.reset()

        self._record_lob(dt, lob, algo)

    def _simulate_to_next_order(self,algo):
        """ Gets the next event from the benchmark algorithm and simulates the LOB up to this point if there are
         remaining orders.
         """

        # get info from the algo about the type and time of next event
        event, done = algo.get_next_event()

        if len(self.remaining_order['benchmark_algo'])!=0 or len(self.remaining_order['rl_algo'])!=0:
            # If we have remaining orders go through the LOBs until they are executed
            while len(self.remaining_order['benchmark_algo'])!=0 or len(self.remaining_order['rl_algo'])!=0:
                # Loop through the LOBs
                dt, lob = self.data_feed.next_lob_snapshot()
                self._record_lob(dt, lob, algo)
                if self.hist_dict['timestamp'][-1] < event['time']:
                    order_temp_bmk, order_temp_rl = self._update_remaining_orders()

                    # place the orders and update the remaining quantities to trade in the algo
                    if type(algo).__name__ != 'RLAlgo':
                        log = self.place_orders(order_temp_bmk,type(algo).__name__)
                        algo.update_remaining_volume(log)
                    else:
                        log = self.place_orders(order_temp_rl,type(algo).__name__)
                        algo.update_remaining_volume(log)

                else:
                    # We have reached the next event with unexecuted volume, if we are not at the end of a bucket
                    # we add it to the volume of next order
                    if event['type'] == 'order_placement':
                        if type(algo).__name__ != 'RLAlgo':
                            unexecuted_vol = self.remaining_order['benchmark_algo'][0]['quantity']
                        else:
                            unexecuted_vol = self.remaining_order['rl_algo'][0]['quantity']

                        algo.volumes_per_trade[algo.bucket_idx][algo.order_idx] += unexecuted_vol

                    # If the event is a bucket end, the market order will be placed according to the bucket_vol_remaining.
                    # Either way, we remove the remaining orders.
                    self.remaining_order['benchmark_algo'] = []
                    self.remaining_order['rl_algo'] = []

        # If we have no remaining orders (for example after executing an entire limit order or after a bucket end),
        # we reset the datafeed to jump to the LOB corresponding to the next event.
        self.data_feed.reset(time='{}:{}:{}'.format(event['time'].hour,event['time'].minute,event['time'].second))
        dt, lob = self.data_feed.next_lob_snapshot()
        self._record_lob(dt, lob, algo)

        return event, done

    def _record_lob(self, dt, lob, algo):
        """ Records lob steps in a dict """

        if type(algo).__name__ != 'RLAlgo':
            self.hist_dict['timestamp'].append(dt)
            self.hist_dict['benchmark_lob'].append(copy.deepcopy(lob))
        else:
            self.hist_dict['timestamp'].append(dt)
            self.hist_dict['rl_lob'].append(copy.deepcopy(lob))

    def _update_remaining_orders(self):
        """ Updates the orders not previously executed with new LOB data """

        order_temp_bmk = None
        order_temp_rl = None
        dt = self.hist_dict['timestamp'][-1]
        if len(self.remaining_order['benchmark_algo']) != 0:
            # update the order and place it
            lob_bmk = self.hist_dict['benchmark_lob'][-1]
            order_temp_bmk = self.remaining_order['benchmark_algo'][0]
            order_temp_bmk['timestamp'] = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f')
            if order_temp_bmk['type'] == 'limit':
                # update the price also
                if order_temp_bmk['side'] == 'bid' and order_temp_bmk['price'] < lob_bmk.get_best_bid():
                    order_temp_bmk['price'] = lob_bmk.get_best_bid() - self.benchmark_algo.tick_size
                if order_temp_bmk['side'] == 'ask' and order_temp_bmk['price'] > lob_bmk.get_best_ask():
                    order_temp_bmk['price'] = lob_bmk.get_best_ask() + self.benchmark_algo.tick_size()
            self.remaining_order['benchmark_algo'] = []

        if len(self.remaining_order['rl_algo']) != 0:
            # update the order and place it
            lob_rl = self.hist_dict['rl_lob'][-1]
            order_temp_rl = self.remaining_order['rl_algo'][0]
            order_temp_rl['timestamp'] = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f')
            if order_temp_rl['type'] == 'limit':
                # update the price also
                if order_temp_rl['side'] == 'bid' and order_temp_rl['price'] < lob_rl.get_best_bid():
                    order_temp_rl['price'] = lob_rl.get_best_bid() - self.benchmark_algo.tick_size
                if order_temp_rl['side'] == 'ask' and order_temp_rl['price'] > lob_rl.get_best_ask():
                    order_temp_rl['price'] = lob_rl.get_best_ask() + self.benchmark_algo.tick_size()
            self.remaining_order['rl_algo'] = []
        return order_temp_bmk, order_temp_rl

    def place_orders(self, order, algo_type):
        """ Places orders of both benchmark and RL algos and store logs in the broker.trade_logs """

        # update the remaining orders
        if algo_type != 'RLAlgo':
            input_lob = copy.deepcopy(self.hist_dict['benchmark_lob'][-1])
            log = place_order(input_lob,
                                  self.hist_dict['timestamp'][-1],
                                  order)
            if log is not None:
                self.trade_logs['benchmark_algo'].append(log)
                bmk_order_temp = order.copy()
                bmk_order_temp['quantity'] -= log['quantity']
                if bmk_order_temp['quantity'] > 0:
                    self.remaining_order['benchmark_algo'].append(bmk_order_temp)
                else:
                    self.remaining_order['benchmark_algo'] = []
            if order['type'] == 'market':
                self.remaining_order['benchmark_algo'] = []

        if algo_type == 'RLAlgo':
            input_lob = copy.deepcopy(self.hist_dict['rl_lob'][-1])
            log = place_order(input_lob,
                                 self.hist_dict['timestamp'][-1],
                                 order)
            if log is not None:
                self.trade_logs['rl_algo'].append(log)
                rl_order_temp = order.copy()
                rl_order_temp['quantity'] -= log['quantity']
                if rl_order_temp['quantity'] > 0:
                    self.remaining_order['rl_algo'].append(rl_order_temp)
                else:
                    self.remaining_order['rl_algo'] = []
            if order['type'] == 'market':
                self.remaining_order['rl_algo'] = []
        return log

    def simulate_algo(self, algo):
        """ Simulates the execution of an algorithm """

        self.reset(algo)
        done = False
        while not done:
            # get next event and order from this
            event, done = self._simulate_to_next_order(algo)
            if type(algo).__name__ != 'RLAlgo':
                algo_order = algo.get_order_at_event(event, self.hist_dict['benchmark_lob'][-1])
            else:
                algo_order = algo.get_order_at_event(event, self.hist_dict['rl_lob'][-1])
            log = self.place_orders(algo_order, type(algo).__name__)

            # update the remaining quantities to trade
            algo.update_remaining_volume(log, event['type'])

    def get_last_bucket_algo(self,algo_type):

        if algo_type=='benchmark':
            bucket_algo = copy.deepcopy(self.benchmark_algo)
            bucket_algo.algo_events = bucket_algo.algo_events[(self.benchmark_algo.no_of_slices+1)*(self.rl_algo.bucket_idx):
                                                                      (self.benchmark_algo.no_of_slices+1)*(self.rl_algo.bucket_idx+1)]
            bucket_algo.bucket_volumes = [bucket_algo.bucket_volumes[self.rl_algo.bucket_idx]]
            bucket_algo.bucket_vol_remaining = [bucket_algo.bucket_vol_remaining[self.rl_algo.bucket_idx]]
            bucket_algo.execution_times = [bucket_algo.execution_times[self.rl_algo.bucket_idx]]
            bucket_algo.volumes_per_trade = [bucket_algo.volumes_per_trade[self.rl_algo.bucket_idx]]
            bucket_algo.volumes_per_trade_default = [bucket_algo.volumes_per_trade_default[self.rl_algo.bucket_idx].copy()]
        elif algo_type == 'rl':
            bucket_algo = copy.deepcopy(self.rl_algo)
            bucket_algo.algo_events = bucket_algo.algo_events[(self.benchmark_algo.no_of_slices+1)*(self.rl_algo.bucket_idx):
                                                              (self.benchmark_algo.no_of_slices+1)*(self.rl_algo.bucket_idx+1)]
            bucket_algo.bucket_volumes = [bucket_algo.bucket_volumes[self.rl_algo.bucket_idx]]
            bucket_algo.bucket_vol_remaining = [bucket_algo.bucket_vol_remaining[self.rl_algo.bucket_idx]]
            bucket_algo.execution_times = [bucket_algo.execution_times[self.rl_algo.bucket_idx]]
            bucket_algo.volumes_per_trade = [bucket_algo.volumes_per_trade[self.rl_algo.bucket_idx]]
            bucket_algo.volumes_per_trade_default = [bucket_algo.volumes_per_trade_default[self.rl_algo.bucket_idx].copy()]

        return bucket_algo

    def calc_vwaps(self):
        """" Calculates the Volume Weighted Average Prices (vwaps) of the Benchmark and RL Algo over the entire episode
        """
        # Check if we have previously calculated the vwaps through calc_vwaps_bucket
        if self.benchmark_algo.bmk_vwap == 0 and self.rl_algo.rl_vwap == 0:
            self.simulate_algo(self.benchmark_algo)
            self.simulate_algo(self.rl_algo)

            bmk_executed_prices = [bmk_trade['quantity']*Decimal(bmk_trade['price'])
                                   for bmk_trade in self.trade_logs['benchmark_algo']
                                   if bmk_trade['message'] == 'trade']
            self.benchmark_algo.bmk_vwap = float(sum(bmk_executed_prices)/self.benchmark_algo.volume)

            rl_executed_prices = [rl_trade['quantity']*Decimal(rl_trade['price'])
                                   for rl_trade in self.trade_logs['rl_algo']
                                   if rl_trade['message'] == 'trade']
            self.rl_algo.rl_vwap = float(sum(rl_executed_prices)/self.rl_algo.volume)

    def calc_vwaps_bucket(self):
        """" Calculates the Volume Weighted Average Prices (vwaps) of the Benchmark and RL Algo over the last bucket
        """
        bucket_bmk_algo = self.get_last_bucket_algo('benchmark')
        self.simulate_algo(bucket_bmk_algo)
        bucket_rl_algo = self.get_last_bucket_algo('rl')
        self.simulate_algo(bucket_rl_algo)

        bmk_executed_prices = [bmk_trade['quantity']*Decimal(bmk_trade['price'])
                               for bmk_trade in self.trade_logs['benchmark_algo']
                               if bmk_trade['message'] == 'trade']

        bmk_bucket_vwap = float(sum(bmk_executed_prices)/(bucket_bmk_algo.bucket_volumes[0] - bucket_bmk_algo.bucket_vol_remaining[0]))

        rl_executed_prices = [rl_trade['quantity']*Decimal(rl_trade['price'])
                               for rl_trade in self.trade_logs['rl_algo']
                               if rl_trade['message'] == 'trade']

        rl_bucket_vwap = float(sum(rl_executed_prices)/(bucket_rl_algo.bucket_volumes[0] - bucket_rl_algo.bucket_vol_remaining[0]))

        unexecuted_rl_vol_percent= float(bucket_rl_algo.bucket_vol_remaining[0])/float(self.rl_algo.bucket_volumes[self.rl_algo.bucket_idx])

        return bmk_bucket_vwap, rl_bucket_vwap, unexecuted_rl_vol_percent

if __name__ == '__main__':

    import random
    import os
    from src.core.environment.limit_orders_setup.execution_algo_real import TWAPAlgo
    from src.data.historical_data_feed import HistoricalDataFeed

    # define the benchmark algo
    algo = TWAPAlgo(trade_direction=1,
                    volume=500,
                    start_time='08:35:05',
                    end_time='09:00:00',
                    no_of_slices=3,
                    bucket_placement_func=lambda no_of_slices: (sorted([round(random.uniform(0, 1), 2) for i in range(no_of_slices)])))

    # define the datafeed
    dir = '/'
    # dir = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution'
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(dir, 'data_dir'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    # define the broker class
    broker = Broker(lob_feed)
    broker.simulate_algo(algo)
    broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])






