import gym
from gym.utils import seeding
from env.orderbook import OrderBook
from env.lob_simulator import LOBObserver
from decimal import Decimal
from env.lob_simulator import split_book_to_orders


class LimitOrderBookEnv(gym.Env):

    def __init__ (self, data_directory, T_Max, past_n=10):

        self.data_directory = data_directory
        self.T_Max = T_Max
        self.observer = LOBObserver(file_name=self.data_directory, past_n=past_n, future_n=T_Max)

        # assume simple action space of placing an order ("0") or not ("1")
        self.action_space = gym.spaces.Discrete(2)

        # Assume observation space takes the first n bid/ask orders as they are
        self.reset()
        obs_space_n = len(self.observer.fut_orders[0]["b"])+len(self.observer.fut_orders[0]["a"])
        self.observation_space = gym.spaces.Discrete(obs_space_n)
        self.seed()

    def step(self, action, extra_data={}):
        """
        Updates the environment given an action.
        """
        assert self.done == False, (
            'reset() must be called before step()')

        observed_lob = self.observer.fut_orders[self.time]
        # assume "silly" bid offer that is at lowest price at the moment (hence probably never traded)
        place_order = {'type' : 'limit',
                       'timestamp': self.time,
                       'side' : 'bid',
                       'quantity' : Decimal(observed_lob["b"][0][1]),
                       'price' : Decimal(observed_lob["b"][0][0]),
                       'trade_id' : 1}
        self.time += 1
        trades, order_id = self.order_book.process_order(place_order, False, False)
        print(self.time)
        self.state = self.observer.fut_orders[self.time]
        self.reward += 1 # assume agent always gets plus 1, no matter what
        self._order_book_step() # update order book to new orders
        if self.time >= self.T_Max - 1:
            self.done = True
        self.info = {}
        return self.state, self.reward, self.done, self.info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """

        # read new data for the Limit-Order-Book
        self.time = 0
        self.observer.draw_random_observations()
        self._order_book_step()
        self.state = self.observer.fut_orders[0]
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def seed(self, seed=None):
        """
        Sets the seed for the environments random number generator.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _order_book_step(self):
        self.order_book = OrderBook()
        _, _, all_orders = split_book_to_orders(self.observer.fut_orders[self.time])
        for order in all_orders:
            _, _ = self.order_book.process_order(order, False, False)
        return self.order_book


if __name__ == '__main__':

    lob_env = LimitOrderBookEnv(data_directory="./data/book_depth_socket_btcusdt_2021_06_21.txt",
                                T_Max=50)
    for t in range(100):
        action = lob_env.action_space.sample()
        observation, reward, done, info = lob_env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    lob_env.close()