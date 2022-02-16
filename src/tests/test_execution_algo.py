import unittest
import numpy as np
from datetime import datetime, timedelta

from src.core.environment.limit_orders_setup.execution_algo_real import TWAPAlgo, BUCKET_SIZES_IN_SECS
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.env_utils import raw_to_order_book


class FakeLOBGenerator:
    fix_lob = np.array([30, 30.1, 30.2,  # asks
                        1, 1, 1,  # ask volumes
                        29.9, 29.8, 29.7,  # bids
                        1, 1, 1])  # bid volumes
    speed_fac = 1
    price_tick = 0.1

    def next_lob_snapshot(self):
        lob_new = self.fix_lob.copy()
        lob_new[:3] -= self.speed_fac * self.price_tick * self.counter
        lob_new[6:9] -= self.speed_fac * self.price_tick * self.counter

        lob_new = np.round(lob_new, 2)

        dt = datetime.today()
        timestamp_dt = datetime(dt.year, dt.month, dt.day, self.time.hour, self.time.minute) + \
                       timedelta(seconds=self.counter)
        self.counter += 1
        lob_out = raw_to_order_book(current_book=lob_new.reshape(-1, 3),
                                    time=timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                    depth=3)
        return timestamp_dt, lob_out

    def reset(self, time):
        try:
            self.time = datetime.strptime(time, '%H:%M:%S.%f')
        except:
            self.time = datetime.strptime(time, '%H:%M:%S')
        self.counter = self.time.second + 1


class TestDataFeed(unittest.TestCase):
    fake_lob = FakeLOBGenerator()
    random_time = "09:00:00"

    def test_feed(self):
        # if the fake feed never gets reset, prices will just go downwards by one tick (0.1) steadily...

        self.fake_lob.reset(time=self.random_time)
        times = []
        best_asks = []
        for i in range(60):
            dt, lob = self.fake_lob.next_lob_snapshot()
            times.append(dt)
            best_asks.append(lob.get_best_ask())
        self.assertEqual(max(np.diff(best_asks)), min(np.diff(best_asks)), 'LOB not growing steadily')


class RLAlgo(TWAPAlgo):
    def __init__(self, *args, **kwargs):
        super(RLAlgo, self).__init__(*args, **kwargs)


class TestExecutionAlgo(unittest.TestCase):
    fake_lob = FakeLOBGenerator()

    # define the benchmark algo
    algo = TWAPAlgo(trade_direction=1,
                    volume=25,
                    start_time='09:00:00',
                    end_time='09:01:00',
                    no_of_slices=1,
                    bucket_placement_func=lambda no_of_slices: 0.5,
                    broker_data_feed=fake_lob)

    # define the broker class
    broker = Broker(fake_lob)
    broker.benchmark_algo = algo
    broker.simulate_algo(algo)

    rl_algo = RLAlgo(trade_direction=1,
                     volume=50,
                     start_time='09:00:00',
                     end_time='09:01:00',
                     no_of_slices=1,
                     bucket_placement_func=lambda no_of_slices: 0.5,
                     broker_data_feed=fake_lob)

    # define the broker class
    broker.rl_algo = rl_algo
    broker.simulate_algo(rl_algo)

    def test_buckets(self):
        # check if we divide this correctly into correct bucket numbers and volume placed

        self.assertEqual(len(self.broker.benchmark_algo.bucket_volumes),
                         np.ceil(60 / BUCKET_SIZES_IN_SECS["1m"]),
                         'Bucket length not as expected')
        self.assertEqual(float(sum(self.broker.benchmark_algo.bucket_volumes)),
                         25,
                         'Total volume is not correct')
        self.assertEqual(float(self.broker.benchmark_algo.bucket_volumes[0]),
                         3,
                         'First bucket volume not correct')

    def test_bug(self):
        # Checks if the collected time stamps in the history are of right lengths (aligned with LOB collection)

        length_bmk_times = len(self.broker.hist_dict['benchmark']['timestamp'])
        length_bmk_lobs = len(self.broker.hist_dict['benchmark']['lob'])
        length_rl_times = len(self.broker.hist_dict['rl']['timestamp'])
        length_rl_lobs = len(self.broker.hist_dict['rl']['lob'])

        # length of timesteps doesnt match anymore with rl and bmk algo...
        self.assertEqual(length_bmk_times, length_bmk_lobs, 'Timestamps and stored LOBs do not match anymore')
        self.assertEqual(length_rl_times, length_rl_lobs, 'Timestamps and stored LOBs do not match anymore')
        self.assertNotEqual(length_bmk_lobs, length_rl_lobs, 'RL history should be longer than Bmk')

    def test_execution(self):
        # check the values of the first execution placed

        self.assertEqual(datetime.strptime(self.broker.trade_logs['benchmark_algo'][0]['timestamp'],
                                           "%Y-%m-%d %H:%M:%S.%f").time().second,
                         4,
                         'First execution time does not seem correct')

        # find first trade...
        all_trades = [log for log in self.broker.trade_logs['benchmark_algo'] if log["message"] == "trade"]
        self.assertEqual(datetime.strptime(all_trades[0]['timestamp'],
                                           "%Y-%m-%d %H:%M:%S.%f").time().second,
                         6,
                         'First execution should be at 6 seconds')
        self.assertEqual(float(all_trades[0]['price']),
                         29.4,
                         'First execution should be at a price of 29.4')
        self.assertEqual(float(all_trades[0]['quantity']),
                         1.0,
                         'First execution should execute a quantity of 1')
        self.assertEqual(float(all_trades[1]['price']),
                         29.35,
                         'Second execution should execute a price of 29.35')

    def test_collection_of_logs(self):
        t_stamps = [datetime.strptime(l["timestamp"], '%Y-%m-%d %H:%M:%S.%f')
                    for l in self.broker.trade_logs['benchmark_algo']]
        t_delta = [(t2 - t1).total_seconds() > 0 for t1, t2 in zip(t_stamps[:-1], t_stamps[1:])]
        self.assertTrue(all(t_delta), 'Time in collected trade logs is not increasing!')

        t_stamps = self.broker.hist_dict['benchmark']["timestamp"]
        t_delta = [(t2 - t1).total_seconds() > 0 for t1, t2 in zip(t_stamps[:-1], t_stamps[1:])]
        self.assertTrue(all(t_delta), 'Time in collected history is not increasing!')

    def test_orders(self):
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][0]['price']),
                         29.4,
                         'First order does not have correct price')
        t = datetime.strptime(self.broker.trade_logs['benchmark_algo'][0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f').time()
        self.assertEqual(t.second, 4, 'First order does not have correct timestamp')

    def test_second_bucket_time(self):
        t_next = datetime.strptime(self.broker.trade_logs['benchmark_algo'][4]['timestamp'],
                                   '%Y-%m-%d %H:%M:%S.%f').time()
        self.assertEqual(t_next.second, 11, 'First order at second bucket is incorrect')

    def test_second_execution(self):
        self.assertEqual(self.broker.trade_logs['benchmark_algo'][3]['message'],
                         'trade',
                         'Second execution should record a trade')
        # since we show volume weighted execution prices, the price should be 29.75
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][3]['price']),
                         29.35,
                         'Second execution price is incorrect')
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][3]['quantity']),
                         2,
                         'Second execution volume is incorrect')

    def test_stored_history(self):
        if len(self.broker.hist_dict['benchmark']['lob']) > 0:
            trades = [len(self.broker.hist_dict['benchmark']['lob'][i].tape)
                      for i in range(len(self.broker.hist_dict['benchmark']['lob']))]
            self.assertNotEqual(sum(trades), 0, 'Benchmark history does not store any trades')
        if len(self.broker.hist_dict['rl']['lob']) > 0:
            trades = [len(self.broker.hist_dict['rl']['lob'][i].tape)
                      for i in range(len(self.broker.hist_dict['rl']['lob']))]
            self.assertNotEqual(sum(trades), 0, 'RL history does not store any trades')


class TestExecutionLogic(unittest.TestCase):

    def test_large_market_order(self):
        """ tests what happens if market orders are so large that they eat into the next order """

        fake_lob = FakeLOBGenerator()

        # define the benchmark algo
        algo = TWAPAlgo(trade_direction=1,
                        volume=1000,
                        start_time='09:00:00',
                        end_time='09:01:00',
                        no_of_slices=1,
                        bucket_placement_func=lambda no_of_slices: 0.99,
                        broker_data_feed=fake_lob)

        # define the broker class
        broker = Broker(fake_lob)
        broker.benchmark_algo = algo
        broker.simulate_algo(algo)

        dts = [trade['timestamp'] for trade in broker.trade_logs['benchmark_algo']]
        res = all(i < j for i, j in zip(dts, dts[1:]))
        self.assertEqual(res, True, 'Overlapping trades detected')

    def test_same_limit_order_timestamps(self):
        """ tests what happens if limit orders are extremely close to each other """

        fake_lob = FakeLOBGenerator()

        # define the benchmark algo
        algo = TWAPAlgo(trade_direction=1,
                        volume=25,
                        start_time='09:00:00',
                        end_time='09:01:00',
                        no_of_slices=2,
                        bucket_placement_func=lambda no_of_slices: [0.5, 0.6],
                        broker_data_feed=fake_lob)

        # define the broker class
        broker = Broker(fake_lob)
        broker.benchmark_algo = algo
        broker.simulate_algo(algo)

        dts = [trade['timestamp'] for trade in broker.trade_logs['benchmark_algo']]
        res = all(i < j for i, j in zip(dts, dts[1:]))
        self.assertEqual(res, True, 'Overlapping trades detected')

    def test_close_limit_orders(self):
        """ tests what happens if limit orders are extremely close to each other """

        fake_lob = FakeLOBGenerator()

        # define the benchmark algo
        algo = TWAPAlgo(trade_direction=1,
                        volume=25,
                        start_time='09:00:00',
                        end_time='09:01:00',
                        no_of_slices=2,
                        bucket_placement_func=lambda no_of_slices: [0.5, 0.8],
                        broker_data_feed=fake_lob)

        # define the broker class
        broker = Broker(fake_lob)
        broker.benchmark_algo = algo
        broker.simulate_algo(algo)

        dts = [trade['timestamp'] for trade in broker.trade_logs['benchmark_algo']]
        res = all(i < j for i, j in zip(dts, dts[1:]))
        self.assertEqual(res, True, 'Overlapping trades detected')

    def test_correct_execution_price(self):

        fake_lob = FakeLOBGenerator()

        # define the benchmark algo
        algo = TWAPAlgo(trade_direction=1,
                        volume=25,
                        start_time='09:00:00',
                        end_time='09:01:00',
                        no_of_slices=1,
                        bucket_placement_func=lambda no_of_slices: 0.5,
                        broker_data_feed=fake_lob)

        # define the broker class
        broker = Broker(fake_lob)
        broker.benchmark_algo = algo
        broker.simulate_algo(algo)

        traded_volumes_per_time = [float(ts['quantity']) for ts in broker.trade_logs['benchmark_algo']
                                   if ts['message'] == 'trade']
        achieved_prices_per_time = [float(ts['price']) for ts in broker.trade_logs['benchmark_algo']
                                    if ts['message'] == 'trade']
        vwap_ours = np.dot(achieved_prices_per_time, traded_volumes_per_time) / sum(traded_volumes_per_time)

        # check that trades in the LOB logs match with our generated logs
        vwap_list = []
        vol_list = []
        for i in range(len(broker.hist_dict['benchmark']['lob'])):
            tp = broker.hist_dict['benchmark']['lob'][i].tape
            if len(tp) > 0:
                vol = 0
                vwap = 0
                for id in range(len(tp)):
                    vol += tp[id]['quantity']
                    vwap += tp[id]['quantity'] * tp[id]['price']
                vwap = float(vwap) / float(vol)
                vwap_list.append(vwap)
                vol_list.append(float(vol))
        vwap_derived = np.dot(vwap_list, vol_list) / sum(vol_list)
        # self.assertEqual(vwap_ours, 27.125999999999998, 'VWAP is not correct anymore')
        self.assertEqual(vwap_ours, vwap_derived, 'VWAPS are not matching')


if __name__ == '__main__':
    unittest.main()
