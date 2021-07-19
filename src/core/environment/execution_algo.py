from abc import ABC, abstractmethod
from decimal import Decimal


class ExecutionAlgo(ABC):

    @abstractmethod
    def __init__(self, trade_direction, quantity, time_steps):
        self.trade_direction = trade_direction
        self.quantity = quantity
        self.time_steps = time_steps
        self.placed_orders = []

    @abstractmethod
    def get_order_at_time(self, step):
        NotImplementedError


class TWAPAlgo(ExecutionAlgo):

    def __init__(self, trade_direction, quantity, time_steps):
        super(TWAPAlgo, self).__init__(trade_direction, quantity, time_steps)
        self.volume_at_time = self.quantity / self.time_steps

    def get_order_at_time(self, step, trade_id=None):
        if trade_id is None:
            trade_id = 1
        if self.trade_direction == 1:
            side = 'bid'
        else:
            side = 'ask'
        place_order = {'type': 'market',
                       'timestamp': step,
                       'side': side,
                       'quantity': Decimal(self.volume_at_time),
                       'trade_id': trade_id}
        self.placed_orders.append(place_order)
        return place_order

