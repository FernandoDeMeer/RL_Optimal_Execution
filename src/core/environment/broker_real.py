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
            trades, _ = lob.process_order(ord, False, False)
        else:
            trades, _ = lob.process_order(ord, True, False)
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
        if type(benchmark_algo).__name__ is not 'RLAlgo':
            self.benchmark_algo.reset(dt.date(), start_time=start_time, end_time=end_time, tick_size=tick)
        self._record_lob(dt, lob)

    def _simulate_to_next_order(self):
        """ Gets the next event from the benchmark algorithm and simulates the LOB up to this point if there are
         remaining orders.
         """

        # get info from the algo about the type and time of next event
        event, done = self.benchmark_algo.get_next_event()

        if len(self.remaining_order['benchmark_algo'])!=0 or len(self.remaining_order['rl_algo'])!=0:
            # If we have remaining orders go through the LOBs until they are executed
            while len(self.remaining_order['benchmark_algo'])!=0 or len(self.remaining_order['rl_algo'])!=0:
                # Loop through the LOBs
                dt, lob = self.data_feed.next_lob_snapshot()
                self._record_lob(dt, lob)
                if self.hist_dict['timestamp'][-1] < event['time']:
                    order_temp_bmk, order_temp_rl = self._update_remaining_orders()
                    bmk_log, _ = self.place_orders(order_temp_bmk, order_temp_rl)

                    # update the remaining quantities to trade in the benchmark algo
                    self.benchmark_algo.update_remaining_volume(bmk_log)
                else:
                    # We have reached the next event with unexecuted volume, if we are not at the end of a bucket
                    # we add it to the volume of next order
                    if event['type'] == 'order_placement':
                        bmk_unexecuted_vol = self.remaining_order['benchmark_algo'][0]['quantity']
                        # rl_unexecuted_vol = self.remaining_order['rl_algo'][0]['quantity']

                        self.benchmark_algo.volumes_per_trade[self.benchmark_algo.bucket_idx][self.benchmark_algo.order_idx] += bmk_unexecuted_vol
                        # self.rl_algo.volumes_per_trade[self.rl_algo.bucket_idx][self.rl_algo.order_idx] += rl_unexecuted_vol

                    # If the event is a bucket end, the market order will be placed according to the bucket_vol_remaining.
                    # Either way, we remove the remaining orders.
                    self.remaining_order['benchmark_algo'] = []
                    self.remaining_order['rl_algo'] = []

        # If we have no remaining orders (for example after executing an entire limit order or after a bucket end),
        # we reset the datafeed to jump to the LOB corresponding to the next event.
        self.data_feed.reset(time='{}:{}:{}'.format(event['time'].hour,event['time'].minute,event['time'].second))
        dt, lob = self.data_feed.next_lob_snapshot()
        self._record_lob(dt, lob)

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

            # bmk_order_temp = benchmark_order
            # bmk_order_temp = benchmark_order.copy()
            benchmark_order['quantity'] -= bmk_log['quantity']
            if benchmark_order['quantity'] > 0:
                self.remaining_order['benchmark_algo'].append(benchmark_order)
            else:
                self.remaining_order['benchmark_algo'] = []
            if benchmark_order['type'] == 'market':
                self.remaining_order['benchmark_algo'] = []

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
            else:
                self.remaining_order['rl_algo'] = []
            if rl_order_temp['type'] == 'market':
                self.remaining_order['rl_algo'] = []
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
            self.benchmark_algo.update_remaining_volume(bmk_log, event['type'])


if __name__ == '__main__':

    import random
    import os
    from src.core.environment.execution_algo_real import TWAPAlgo
    from src.data.historical_data_feed import HistoricalDataFeed

    # define the benchmark algo
    algo = TWAPAlgo(trade_direction=1,
                    volume=500,
                    start_time='08:35:05',
                    end_time='09:00:00',
                    no_of_slices=3,
                    bucket_placement_func=lambda no_of_slices: (sorted([round(random.uniform(0, 1), 2) for i in range(no_of_slices)])))

    # define the datafeed
    dir = 'C:\\Users\\demp\\Documents\\Repos\\RLOptimalTradeExecution'
    # dir = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution'
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(dir, 'data_dir'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    # define the broker class
    broker = Broker(lob_feed, use_for_rl=False)
    broker.simulate_algo(algo)
    broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])


    # ToDo:
    # Debugging:
    #   hist_dict has gaps when trades at events are placed!
    #   * cancel remaining orders upon bucket end -- DONE (lines 157-18 175-176)
    #   * possibly introduce a max depth to be able to trade in the order book
    #   * too slow, no need to loop through the entire thing if no trades remaining!! -- DONE (func _update_remaining_orders)
    #   * Bugs: plot: fix time axis
    #   * migrate this into a new env
    #   * implement an initialisation class that draws randomly from starting points -- Now done in the BaseEnv, maybe make it a function of the ExecutionAlgo base class?






