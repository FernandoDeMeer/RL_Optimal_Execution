import copy
import numpy as np
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
            # lob_temp = copy.deepcopy(lob)
            trades, _ = lob.process_order(ord, False, False)
        else:
            # lob_temp = copy.deepcopy(lob)
            trades, _ = lob.process_order(ord, True, False)
        if trades:
            vol_wgt_price, vol = calc_volume_weighted_price_from_trades(trades)
            msg = 'trade'
        else:
            vol_wgt_price, vol, msg = order['price'], 0, 'no_trade'
        trade_message = {'timestamp': datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f'),
                         'message': msg,
                         'type': order['type'],
                         'side': order['side'],
                         'price': Decimal(str(vol_wgt_price)),
                         'quantity': Decimal(str(vol)),
                         'target_quantity': order['quantity']}
    return trade_message


class Broker(ABC):
    """ Currently only for placing trades and getting volume weighted execution prices """

    def __init__(self, data_feed,):

        self.data_feed = data_feed
        self.benchmark_algo = None
        self.rl_algo = None
        self.hist_dict = {'benchmark': {'timestamp': [], 'lob': []},
                          'rl': {'timestamp': [], 'lob': []}}
        self.remaining_order = {'benchmark_algo': [],
                                'rl_algo': []}
        self.trade_logs = {'benchmark_algo': [],
                           'rl_algo': []}

    def reset(self, algo):
        """ Resetting the Broker class """

        self.data_feed.reset(time=algo.start_time)
        dt, lob = self.data_feed.next_lob_snapshot()

        # reset the Broker logs
        if type(algo).__name__ != 'RLAlgo':
            self.hist_dict['benchmark']['timestamp'] = []
            self.hist_dict['benchmark']['lob'] = []
            self.remaining_order['benchmark_algo'] = []
            self.trade_logs['benchmark_algo']= []
            self.current_dt_bmk = dt
        else:
            self.hist_dict['rl']['timestamp'] = []
            self.hist_dict['rl']['lob'] = []
            self.remaining_order['rl_algo'] = []
            self.trade_logs['rl_algo'] = []
            self.current_dt_rl = dt

        # update to the first instance of the datafeed & record this
        algo.reset()
        # self._record_lob(dt, lob, algo)

    def simulate_algo(self, algo):
        """ Simulates the execution of an algorithm """

        self.reset(algo)
        # simulate to first event...
        event, done, lob = self.simulate_to_next_event(algo)
        while not done:

            # place order at event...
            _ = self.place_next_order(algo, event, done, lob)

            # simulate to next event...
            event, done, lob = self.simulate_to_next_event(algo)

            # if event is a bucket bound, we place another trade (if necessary)...
            if event['type'] == 'bucket_bound':
                done = self.place_next_order(algo, event, done, lob)
            if not done:
                # if still not done, then simulate again to next event...
                event, done, lob = self.simulate_to_next_event(algo)

    def simulate_to_next_event(self, algo):
        """ Gets the next event from the benchmark algorithm and simulates the LOB up to this point if there are
         remaining orders. This does not actively place trades, but simulates trades that remain in the market.

         """

        # get info from the algo about the type and time of next event
        event, done = algo.get_next_event()

        if type(algo).__name__ != 'RLAlgo':
            remaining_order = self.remaining_order['benchmark_algo']
            self.data_feed.reset(time=self.current_dt_bmk.time().strftime('%H:%M:%S.%f'))
        else:
            remaining_order = self.remaining_order['rl_algo']
            self.data_feed.reset(time=self.current_dt_rl.time().strftime('%H:%M:%S.%f'))

        if len(remaining_order) != 0:
            # If we have remaining orders go through the LOBs until they are executed
            while len(remaining_order) != 0:
                # Loop through the LOBs
                dt, lob = self.data_feed.next_lob_snapshot()
                self._record_lob(dt, lob, algo)
                if dt <= event['time']:
                    order_temp_bmk, order_temp_rl = self._update_remaining_orders()

                    # place the orders and update the remaining quantities to trade in the algo
                    if type(algo).__name__ != 'RLAlgo':
                        log = self.place_orders(order_temp_bmk, type(algo).__name__)
                        algo.update_remaining_volume(log)
                        remaining_order = self.remaining_order['benchmark_algo']
                    else:
                        log = self.place_orders(order_temp_rl, type(algo).__name__)
                        algo.update_remaining_volume(log)
                        remaining_order = self.remaining_order['rl_algo']
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
                    if type(algo).__name__ != 'RLAlgo':
                        self.remaining_order['benchmark_algo'] = []
                    else:
                        self.remaining_order['rl_algo'] = []
                    remaining_order = []

        # If we have no remaining orders (for example after executing an entire limit order or after a bucket end),
        # we reset the datafeed to jump to the LOB corresponding to the next event.
        self.data_feed.reset(time=event['time'].time().strftime('%H:%M:%S.%f'))
        dt, lob = self.data_feed.next_lob_snapshot()
        self._record_lob(dt, lob, algo)
        if type(algo).__name__ != 'RLAlgo':
            self.current_dt_bmk = dt
        else:
            self.current_dt_rl = dt
        return event, done, lob

    def place_next_order(self, algo, event, done, lob, vol=None):

        if type(algo).__name__ != 'RLAlgo':
            self.data_feed.reset(time=self.current_dt_bmk.time().strftime('%H:%M:%S.%f'))
        else:
            self.data_feed.reset(time=self.current_dt_rl.time().strftime('%H:%M:%S.%f'))

        algo_order = algo.get_order_at_event(event, lob)
        if vol is not None:
            algo_order['quantity'] = Decimal(str(vol))
        log = self.place_orders(algo_order, type(algo).__name__)

        # update the remaining quantities to trade
        algo.update_remaining_volume(log, event['type'])

        if len(self.remaining_order['benchmark_algo']) != 0:
            if self.remaining_order['benchmark_algo'][0]['type'] == 'market':
                # We have a market order that didn't fully execute, so we place it again on subsequent LOBs until it is fully executed.
                while len(self.remaining_order['benchmark_algo'])!= 0:
                    dt, lob = self.data_feed.next_lob_snapshot()
                    self._record_lob(dt, lob, algo)
                    order_temp_bmk, order_temp_rl = self._update_remaining_orders()
                    # place the orders and update the remaining quantities to trade in the algo
                    log = self.place_orders(order_temp_bmk,type(algo).__name__)
                    algo.vol_remaining -= Decimal(str(log['quantity']))
                    algo.bucket_vol_remaining[algo.bucket_idx-1] -= Decimal(str(log['quantity']))
                    self.current_dt_bmk = dt
        if len(self.remaining_order['rl_algo'])!= 0:
            if self.remaining_order['rl_algo'][0]['type'] == 'market':
                # We have a market order that didn't fully execute, so we place it again on subsequent LOBs until it is fully executed.
                while len(self.remaining_order['rl_algo'])!= 0:
                    dt, lob = self.data_feed.next_lob_snapshot()
                    self._record_lob(dt, lob, algo)
                    order_temp_bmk, order_temp_rl = self._update_remaining_orders()
                    # place the orders and update the remaining quantities to trade in the algo
                    log = self.place_orders(order_temp_rl,type(algo).__name__)
                    algo.vol_remaining -= Decimal(str(log['quantity']))
                    algo.bucket_vol_remaining[algo.bucket_idx-1] -= Decimal(str(log['quantity']))
                    self.current_dt_rl = dt

        if type(algo).__name__ != 'RLAlgo':
            self.benchmark_algo = algo
        else:
            self.rl_algo = algo
        return done

    def _record_lob(self, dt, lob, algo):
        """ Records lob steps in a dict """

        if type(algo).__name__ != 'RLAlgo':
            self.hist_dict['benchmark']['timestamp'].append(dt)
            self.hist_dict['benchmark']['lob'].append(copy.deepcopy(lob))
        else:
            self.hist_dict['rl']['timestamp'].append(dt)
            self.hist_dict['rl']['lob'].append(copy.deepcopy(lob))

    def _update_remaining_orders(self):
        """ Updates the orders not previously executed with new LOB data """

        order_temp_bmk = None
        order_temp_rl = None

        if len(self.remaining_order['benchmark_algo']) != 0:
            # update the order and place it
            dt = self.hist_dict['benchmark']['timestamp'][-1]
            lob_bmk = self.hist_dict['benchmark']['lob'][-1]
            order_temp_bmk = self.remaining_order['benchmark_algo'][0]
            order_temp_bmk['timestamp'] = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f')
            if order_temp_bmk['type'] == 'limit':
                # update the price also
                if order_temp_bmk['side'] == 'bid' and order_temp_bmk['price'] < lob_bmk.get_best_bid():
                    order_temp_bmk['price'] = lob_bmk.get_best_bid() - self.benchmark_algo.tick_size
                if order_temp_bmk['side'] == 'ask' and order_temp_bmk['price'] > lob_bmk.get_best_ask():
                    order_temp_bmk['price'] = lob_bmk.get_best_ask() + self.benchmark_algo.tick_size
            self.remaining_order['benchmark_algo'] = []

        if len(self.remaining_order['rl_algo']) != 0:
            # update the order and place it
            dt = self.hist_dict['rl']['timestamp'][-1]
            lob_rl = self.hist_dict['rl']['lob'][-1]
            order_temp_rl = self.remaining_order['rl_algo'][0]
            order_temp_rl['timestamp'] = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S.%f')
            if order_temp_rl['type'] == 'limit':
                # update the price also
                if order_temp_rl['side'] == 'bid' and order_temp_rl['price'] < lob_rl.get_best_bid():
                    order_temp_rl['price'] = lob_rl.get_best_bid() - self.benchmark_algo.tick_size
                if order_temp_rl['side'] == 'ask' and order_temp_rl['price'] > lob_rl.get_best_ask():
                    order_temp_rl['price'] = lob_rl.get_best_ask() + self.benchmark_algo.tick_size
            self.remaining_order['rl_algo'] = []
        return order_temp_bmk, order_temp_rl

    def place_orders(self, order, algo_type):
        """ Places orders of both benchmark and RL algos and store logs in the broker.trade_logs """

        # update the remaining orders
        if algo_type != 'RLAlgo':
            log = place_order(self.hist_dict['benchmark']['lob'][-1],
                              self.hist_dict['benchmark']['timestamp'][-1],
                              order)
            if log is not None:
                self.trade_logs['benchmark_algo'].append(log)
                bmk_order_temp = order.copy()
                bmk_order_temp['quantity'] -= log['quantity']
                if bmk_order_temp['quantity'] > 0:
                    self.remaining_order['benchmark_algo'].append(bmk_order_temp)
                else:
                    self.remaining_order['benchmark_algo'] = []

        if algo_type == 'RLAlgo':
            log = place_order(self.hist_dict['rl']['lob'][-1],
                              self.hist_dict['rl']['timestamp'][-1],
                              order)
            if log is not None:
                self.trade_logs['rl_algo'].append(log)
                rl_order_temp = order.copy()
                rl_order_temp['quantity'] -= log['quantity']
                if rl_order_temp['quantity'] > 0:
                    self.remaining_order['rl_algo'].append(rl_order_temp)
                else:
                    self.remaining_order['rl_algo'] = []
        return log

    def calc_vwap_from_logs(self, start_date=None, end_date=None):

        f = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        # filter by start idx
        if start_date is None:
            start_idx_bmk = 0
            start_idx_rl = 0
        else:
            start_idx_bmk = next(x[0] for x in enumerate([f(log["timestamp"]) for log in self.trade_logs['benchmark_algo']])
                                 if x[1] > start_date)
            start_idx_rl = next(x[0] for x in enumerate([f(log["timestamp"]) for log in self.trade_logs['rl_algo']])
                                if x[1] > start_date)

        # filter by end idx
        if end_date is None:
            end_idx_bmk = len(self.trade_logs['benchmark_algo'])
            end_idx_rl = len(self.trade_logs['rl_algo'])
        else:
            try:
                end_idx_bmk = next(x[0] for x in enumerate([f(log["timestamp"]) for log in self.trade_logs['benchmark_algo']])
                                   if x[1] >= end_date) + 1
            except:
                end_idx_bmk = len(self.trade_logs['benchmark_algo'])
            try:
                end_idx_rl = next(x[0] for x in enumerate([f(log["timestamp"]) for log in self.trade_logs['rl_algo']])
                                  if x[1] >= end_date) + 1
            except:
                end_idx_rl = len(self.trade_logs['rl_algo'])

        # get trade logs between the two dates
        if len(self.trade_logs['benchmark_algo']) != 0:
            bmk_vwap = self._calc_vwap(self.trade_logs['benchmark_algo'][start_idx_bmk:end_idx_bmk])
        else:
            bmk_vwap = 0

        if len(self.trade_logs['rl_algo']) != 0:
            rl_vwap = self._calc_vwap(self.trade_logs['rl_algo'][start_idx_rl:end_idx_rl])
        else:
            rl_vwap = 0

        return bmk_vwap, rl_vwap

    @staticmethod
    def _calc_vwap(logs):
        p = [Decimal(trade['price']) for trade in logs if trade['message'] == 'trade']
        v = [trade['quantity'] for trade in logs if trade['message'] == 'trade']
        if len(p) == 0 or len(v) == 0:
            return 0
        vwap = float(np.dot(p, v)/sum(v))
        return vwap
