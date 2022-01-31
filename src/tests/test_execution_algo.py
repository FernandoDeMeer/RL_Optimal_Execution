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

        dt = datetime.today()
        timestamp_dt = datetime(dt.year, dt.month, dt.day, self.time.hour, self.time.minute, self.time.second) + \
                       timedelta(seconds=self.counter)
        self.counter += 1
        lob_out = raw_to_order_book(current_book=lob_new.reshape(-1, 3),
                                    time=timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                    depth=3)
        return timestamp_dt, lob_out

    def reset(self, time):
        self.counter = 0
        self.time = datetime.strptime(time, '%H:%M:%S')


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

    def test_buckets(self):
        # check if we divide this correctly into correct bucket numbers and volume placed
        self.assertEqual(len(self.broker.benchmark_algo.bucket_volumes),
                         np.ceil(60/BUCKET_SIZES_IN_SECS["1m"]),
                         'Bucket length not as expected')
        self.assertEqual(float(sum(self.broker.benchmark_algo.bucket_volumes)),
                         25,
                         'Total volume is not correct')
        self.assertEqual(float(self.broker.benchmark_algo.bucket_volumes[0]),
                         3,
                         'First bucket volume not correct')

    def test_first_order(self):
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][0]['price']),
                         29.8,
                         'First order does not have correct price')
        t = datetime.strptime(self.broker.trade_logs['benchmark_algo'][0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f').time()
        self.assertEqual(t.second, 3, 'First order does not have correct timestamp')

    def test_first_execution(self):
        self.assertEqual(self.broker.trade_logs['benchmark_algo'][2]['message'],
                         'trade',
                         'First execution should record a trade')
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][2]['price']),
                         29.8,
                         'Incorrect price at first execution')
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][2]['quantity']),
                         1.0,
                         'Incorrect volume at first execution')

    def test_second_execution(self):
        self.assertEqual(self.broker.trade_logs['benchmark_algo'][3]['message'],
                         'trade',
                         'Second execution should record a trade')
        # since we show volume weighted execution prices, the price should be 29.75
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][3]['price']),
                         29.75,
                         'Second execution price is incorrect')
        self.assertEqual(float(self.broker.trade_logs['benchmark_algo'][3]['quantity']),
                         2,
                         'Second execution volume is incorrect')

    def test_second_bucket_time(self):
        t_next = datetime.strptime(self.broker.trade_logs['benchmark_algo'][4]['timestamp'],
                                   '%Y-%m-%d %H:%M:%S.%f').time()
        self.assertEqual(t_next.second, 10, 'First order at second bucket is incorrect')


if __name__ == '__main__':
    unittest.main()

    # plot the executed trades against the trades from the original schedule...
    # broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])
