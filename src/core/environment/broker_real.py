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

    trade_message = None
    if order['quantity'] > 0:
        if order['type'] == 'limit':
            trades, _ = lob.process_order(order, False, False)
        else:
            trades, _ = lob.process_order(order, True, False)
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

    def __init__(self, data_feed, use_for_rl=True):

        self.data_feed = data_feed
        self.benchmark_algo = None
        self.use_for_rl = use_for_rl

    def reset(self, benchmark_algo, start_time=None, end_time=None):
        """ Resetting the Broker class """

        # reset the new benchmark algorithm
        self.benchmark_algo = benchmark_algo

        self.hist_dict = {'timestamp': [],
                          'benchmark_lob': [],
                          'rl_lob': []}
        self.remaining_order = {'benchmark_algo': [],
                                'rl_algo': []}
        self.trade_logs = {'benchmark_algo': [],
                           'rl_algo': []}

        # update to the first instance of the datafeed & record this
        self.data_feed.reset(time=benchmark_algo.start_time)
        dt, lob = self.data_feed.next_lob_snapshot()

        # get the tick size implied by LOB data
        v = lob.bids.get_price_list(lob.get_best_bid()).volume
        tick = Decimal(str(1 / (10 ** abs(v.as_tuple().exponent))))
        self.benchmark_algo.reset(dt.date(), start_time=start_time, end_time=end_time, tick_size=tick)
        self._record_lob(dt, lob)

    def _simulate_to_next_order(self):
        """ Gets the next event from the benchmark algorithm and simulates the LOB up to this point """

        # get info from the algo about the type and time of next event
        event, done = self.benchmark_algo.get_next_event()

        while self.hist_dict['timestamp'][-1] < event['time']:
            # just update the LOB history if nothing happens
            dt, lob = self.data_feed.next_lob_snapshot()
            self._record_lob(dt, lob)
            order_temp_bmk, order_temp_rl = self._update_remaining_orders()
            bmk_log, _ = self.place_orders(order_temp_bmk, order_temp_rl)

            # update the remaining quantities to trade in the benchmark algo
            if bmk_log is not None and bmk_log['quantity'] > 0:
                print(bmk_log['quantity'])
                self.benchmark_algo.update_remaining_volume(bmk_log['quantity'])

        return event, done

    def _record_lob(self, dt, lob):
        """ Records lob steps in a dict """

        self.hist_dict['timestamp'].append(dt)
        self.hist_dict['benchmark_lob'].append(copy.deepcopy(lob))
        if self.use_for_rl:
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
                    order_temp_bmk['price'] = lob_bmk.get_best_bid() - self.benchmark_algo.tick_size # Don't orders not previously executed already have a set price? This updates their price on every new lob
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

    def place_orders(self, benchmark_order, rl_order=None):
        """ Places orders of both benchmark and RL algos """

        # update the remaining orders
        bmk_log, rl_log = None, None
        if benchmark_order is not None:

            bmk_log = place_order(self.hist_dict['benchmark_lob'][-1],
                                  self.hist_dict['timestamp'][-1],
                                  benchmark_order)
            if bmk_log is not None:
                self.trade_logs['benchmark_algo'].append(bmk_log)

            bmk_order_temp = benchmark_order.copy()
            bmk_order_temp['quantity'] -= bmk_log['quantity']
            if bmk_order_temp['quantity'] > 0:
                self.remaining_order['benchmark_algo'].append(bmk_order_temp)

        if rl_order is not None:

            if self.use_for_rl:
                rl_log = place_order(self.hist_dict['rl_lob'][-1],
                                     self.hist_dict['timestamp'][-1],
                                     rl_order)
                if rl_log is not None:
                    self.trade_logs['rl_algo'].append(rl_log)

            rl_order_temp = rl_order.copy()
            rl_order_temp['quantity'] -= rl_log['quantity']
            if rl_order_temp['quantity'] > 0:
                self.remaining_order['rl_algo'].append(rl_order_temp)
        return bmk_log, rl_log

    def simulate_algo(self, algo):
        """ Simulates the execution of the benchmark algorithm """

        self.reset(algo)
        done = False
        while not done:
            # get next event and order from this
            event, done = self._simulate_to_next_order()
            algo_order = self.benchmark_algo.get_order_at_event(event, self.hist_dict['benchmark_lob'][-1])
            bmk_log, _ = self.place_orders(algo_order)

            # update the remaining quantities to trade
            if bmk_log is not None and bmk_log['quantity'] > 0:
                print(bmk_log['quantity'])
                self.benchmark_algo.update_remaining_volume(bmk_log['quantity'])


if __name__ == '__main__':

    import random
    import os
    from src.core.environment.execution_algo_real import TWAPAlgo
    from src.data.historical_data_feed import HistoricalDataFeed

    # define the benchmark algo
    algo = TWAPAlgo(trade_direction=1,
                    volume=500,
                    start_time='08:45:00',
                    end_time='09:00:00',
                    no_of_slices=3,
                    bucket_placement_func=lambda no_of_slices: (sorted([round(random.uniform(0, 1), 2) for i in range(no_of_slices)])))

    # define the datafeed
    dir = 'C:\\Users\\demp\\Documents\\Repos\\RLOptimalTradeExecution'
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(dir, 'data_dir'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    # define the broker class
    broker = Broker(lob_feed, use_for_rl=False)
    broker.simulate_algo(algo)
    broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])


    # ToDo:
    #   * cancel remaining orders upon bucket end
    #   * possibly introduce a max depth to be able to trade in the order book
    #   * too slow, no need to loop through the entire thing if no trades remaining!!
    #   * Bugs: plot: fix time axis
    #   * discuss with others how to make this faster
    #   * migrate this into a new env
    #   * implement an initialisation class that draws randomly from starting points






