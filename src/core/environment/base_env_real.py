import gym
import random
import numpy as np
from datetime import datetime,timedelta
from gym.utils import seeding
from decimal import Decimal
from abc import ABC
from src.core.environment.execution_algo_real import ExecutionAlgo,TWAPAlgo,RLAlgo
from src.core.environment.broker_real import Broker
from src.ui.user_interface import UIAppWindow, UserInterface
import time
import copy


def lob_to_numpy(lob, depth, norm_price=None, norm_vol_bid=None, norm_vol_ask=None):
    bid_prices = lob.bids.prices[-depth:]
    bid_volumes = [float(lob.bids.get_price_list(p).volume) for p in bid_prices]
    bid_prices = [float(bids) for bids in bid_prices]
    ask_prices = lob.asks.prices[:depth]
    ask_volumes = [float(lob.asks.get_price_list(p).volume) for p in ask_prices]
    ask_prices = [float(asks) for asks in ask_prices]

    if norm_price:
        prices = (np.array(bid_prices + ask_prices) / float(norm_price)-1)*100 #have to make sure bid_prices and ask_prices are lists
    else:
        prices = np.array(bid_prices + ask_prices)

    if norm_vol_bid and norm_vol_ask:
        volumes = np.concatenate((np.array(bid_volumes)/float(norm_vol_bid),
                            np.array(ask_volumes)/float(norm_vol_ask)), axis=0)
    else:
        volumes = np.concatenate((np.array(bid_volumes),
                                  np.array(ask_volumes)), axis=0)
    return np.concatenate((prices, volumes), axis=0)


class BaseEnv(gym.Env, ABC):

    def __init__(self,
                 show_ui,
                 broker,
                 obs_config,
                 action_space,
                 ):

        if show_ui:
            self.ui = UIAppWindow()
        else:
            self.ui = None

        self.broker = broker
        self.obs_config = obs_config
        self.reset()
        self.build_observation_space()
        self.action_space = action_space
        self.seed()

    def reset(self):
        # Randomize the volume, no_of_slices, start_time, end_time and rand_bucket_bounds of self.benchmark_algo
        start_time = '{}:{}:{}'.format(random.randint(0,23),
                                       random.randint(0,59),
                                       random.randint(0,59))
        start_time = str(datetime.strptime(start_time,'%H:%M:%S').time())
        execution_horizon = datetime.strptime('{}:{}:{}'.format(0,
                                                                random.randint(0,59),
                                                                random.randint(0,59)),'%H:%M:%S').time() #Set execution horizon within the minute scale
        self.broker.benchmark_algo = TWAPAlgo(trade_direction= 1,  # Train different models for each trade_direction
                                            volume= random.randint(500,1000),
                                            no_of_slices= random.randint(3,10),
                                            bucket_placement_func=lambda no_of_slices: (sorted([round(random.uniform(0, 1), 2) for i in range(no_of_slices)])),
                                            start_time= start_time,
                                            end_time= str((datetime.strptime(start_time,'%H:%M:%S') + timedelta(hours= execution_horizon.hour,minutes=execution_horizon.minute,seconds=execution_horizon.second)).time()),
                                            rand_bucket_bounds_width= random.randint(10, 20),  # % of the bucket_size
                                            broker_data_feed= self.broker.data_feed)
        # reset the broker with the new benchmark_algo

        self.broker.reset(self.broker.benchmark_algo,)

        # simulate the benchmark algo and declare the RLAlgo
        self.broker.simulate_algo(self.broker.benchmark_algo)

        self.broker.rl_algo = RLAlgo(benchmark_algo= self.broker.benchmark_algo,
                                 trade_direction=self.broker.benchmark_algo.trade_direction,
                                 volume= self.broker.benchmark_algo.volume,
                                 no_of_slices= self.broker.benchmark_algo.no_of_slices,
                                 bucket_placement_func= self.broker.benchmark_algo.bucket_placement_func,
                                 broker_data_feed=self.broker.data_feed)

        self.mid_price_history = []

        # To build the first observation we need to reset the datafeed to the timestamp of the first algo_event

        self.state = self.build_observation_at_event(event_time = self.broker.benchmark_algo.algo_events[0].strftime('%H:%M:%S'))
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def step(self, action):

        # print("time idx: {}\t action: {}".format(self.time, action))

        assert self.done is False, (
            'reset() must be called before step()')

        self.broker.rl_algo.event_idx += 1

        #Update the volume the RLAlgo has chosen for the order
        vol_to_trade = Decimal(str(action[0])) * self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx].quantize(Decimal('1.0'))
        self.broker.rl_algo.volumes_per_trade[self.broker.rl_algo.bucket_idx][self.broker.rl_algo.order_idx] = vol_to_trade
        self.broker.rl_algo.order_idx += 1

        # update the remaining bucket vol
        self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx] -= vol_to_trade

        if self.broker.rl_algo.event_idx % self.broker.rl_algo.no_of_slices == 0:
            self.broker.rl_algo.bucket_idx += 1
            self.broker.rl_algo.order_idx = 0


    # Calculate rewards, if we are at the end of an episode
        if self.broker.rl_algo.event_idx >= len(self.broker.rl_algo.algo_events) - self.broker.rl_algo.buckets.n_buckets:
            self.calc_reward()
            self.done = True
            self.state = []
        else:
            # Go to the next event otherwise
            self.state = self.build_observation_at_event(event_time = self.broker.rl_algo.algo_events[self.broker.rl_algo.event_idx].strftime('%H:%M:%S'))

        self.info = {}
        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):

        self.ui.user_interface.update_data([
            {
                "event": "{}#{}".format(UserInterface.CHART_0, "0"),
                "data": np.array(copy.deepcopy(self.benchmark_qty_remaining_history))
            },
            {
                "event": "{}#{}".format(UserInterface.CHART_0, "1"),
                "data": np.array(copy.deepcopy(self.rl_qty_remaining_history))
            },
            {
                "event": "{}#{}".format(UserInterface.CHART_1, "0"),
                "data": np.array(copy.deepcopy(self.trades_monitor.data["benchmark"]["qty"]))
            },
            {
                "event": "{}#{}".format(UserInterface.CHART_1, "1"),
                "data": np.array(copy.deepcopy(self.trades_monitor.data["rl"]["qty"]))
            },
            {
                "event": "{}#{}".format(UserInterface.CHART_2, "0"),
                "data": np.array(copy.deepcopy(self.mid_price_history))
            },
        ])

        time.sleep(0.1)
        self.ui.controller.check_wait_on_event("wait_step")

        if self.done:
            print("done")
            self.ui.controller.check_wait_on_event("wait_episode")

    def build_observation_space(self):

        """
        Observation Space Config Parameters

        nr_of_lobs : int, Number of past snapshots to be concatenated to the latest snapshot
        lob_depth : int, Depth of the LOB to be in each snapshot (max lob_depth = 20 )
        norm : Boolean, normalize or not -- We take the strike price to normalize with as the middle of the bid/ask spread --
        """

        n_obs_onesided = self.obs_config['lob_depth'] * \
                         self.obs_config['nr_of_lobs']
        zeros = np.zeros(n_obs_onesided)
        ones = np.ones(n_obs_onesided)

        """
            The bounds are as follows (if we allow normalisation of past LOB snapshots by current LOB data):
                Inf > bids_price <= 0,
                Inf > asks_price > 0,
                Inf > bids_volume >= 0,
                Inf > asks_volume >= 0,
                benchmark_algo.volume >= remaining_vol_to_trade >= 0,
                no_of_slices >= remaining_orders >= 0
        """
        low = np.concatenate((zeros, zeros, zeros, zeros, np.array([0]), np.array([0])), axis=0)
        high = np.concatenate((ones*np.inf, ones*np.inf,
                               ones*np.inf, ones*np.inf,
                               np.array([self.broker.benchmark_algo.volume]),
                               np.array([self.broker.benchmark_algo.no_of_slices])), axis= 0)

        obs_space_n = (n_obs_onesided * 4 + 2)
        assert low.shape[0] == high.shape[0] == obs_space_n
        self.observation_space = gym.spaces.Box(low=low,
                                                high=high,
                                                shape=(obs_space_n,),
                                                dtype=np.float64)

    def build_observation_at_event(self, event_time):
        # Build observation using the history of order book data, first reset the data_feed at the event timestamp
        self.broker.data_feed.reset(time=event_time)
        # Sample past lobs
        past_dts, past_lobs = self.broker.data_feed.past_lob_snapshots(no_of_past_lobs=self.obs_config['nr_of_lobs']-1)
        for dt in past_dts:
            self.broker.hist_dict['timestamp'].append(dt)
        for lob in past_lobs:
            self.broker.hist_dict['rl_lob'].append(lob)
        # Sample the current lob
        dt, lob = self.broker.data_feed.next_lob_snapshot()
        self.broker.hist_dict['timestamp'].append(dt)
        self.broker.hist_dict['rl_lob'].append(lob)

        obs = np. array([])
        if self.obs_config['norm']:
            mid = (self.broker.hist_dict['rl_lob'][-1].get_best_ask() +
                   self.broker.hist_dict['rl_lob'][-1].get_best_bid()) / 2
            vol_bid = self.broker.hist_dict['rl_lob'][-1].bids.volume
            vol_ask = self.broker.hist_dict['rl_lob'][-1].asks.volume
            for lob in self.broker.hist_dict['rl_lob'][-self.obs_config['nr_of_lobs']:]:
                #TODO: Right now both prices and volumes are normalized, however we show the agent the unnormalized volume, wouldn't it make more sense to not normalize the volume so that it can find the relationship between those quantities?
                obs = np.concatenate((obs, lob_to_numpy(lob,
                                                   depth=self.obs_config['lob_depth'],
                                                   norm_price=mid,
                                                   norm_vol_bid=None,
                                                   norm_vol_ask=None)), axis=0)
            obs = np.concatenate((obs, np.array([self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]]),#vol left to trade in the bucket
                                  np.array([self.broker.rl_algo.no_of_slices - self.broker.rl_algo.order_idx -1])), axis=0)# orders left to place in the bucket
        else:
            for lob in self.broker.hist_dict['rl_lob'][-self.obs_config['nr_of_lobs']:]:
                obs = np.concatenate(obs, (lob_to_numpy(lob,
                                                        depth=self.obs_config['lob_depth'])), axis=0)
            obs = np.concatenate((obs, np.array([self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]]),#vol left to trade in the bucket
                                  np.array([self.broker.rl_algo.no_of_slices - self.broker.rl_algo.order_idx -1])), axis=0)# orders left to place in the bucket

        # need to make sure that obs fits to the observation space...
        # 0 padding whenever this gets smaller...
        # NaN in the beginning if I don't have history yet...

        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calc_reward(self,):
        bmk_vwap, rl_vwap = self.broker.calc_vwaps()
        if (rl_vwap - bmk_vwap) * self.broker.rl_algo.trade_direction < 0:
            self.reward += 1


if __name__ == '__main__':

    import random
    import os
    from src.core.environment.execution_algo_real import TWAPAlgo
    from src.data.historical_data_feed import HistoricalDataFeed


    # define the datafeed
    dir = 'C:\\Users\\demp\\Documents\\Repos\\RLOptimalTradeExecution'
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(dir, 'data_dir'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    # define the broker class
    broker = Broker(lob_feed)

    # for i in range(100):
    #     t = time.time()
    #     broker.simulate_algo(algo)
    #     broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])
    #     elapsed = time.time() - t
    #     print(elapsed)

    # define the obs_config
    observation_space_config = {'lob_depth': 5, 'nr_of_lobs': 5, 'norm': True}
    # define action space
    action_space = gym.spaces.Box(low=0.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)
    # define the env
    base_env = BaseEnv(show_ui=False, broker=broker,obs_config= observation_space_config, action_space = action_space)
    for i in range(len(base_env.broker.rl_algo.algo_events) + 1):
        base_env.step(action = np.array([0.5]))
        if base_env.done == True:
            base_env.reset()