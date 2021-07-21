import copy
import numpy as np
from abc import ABC


def calc_volume_weighted_price_from_trades(trades):
    volume = 0
    price = 0
    for idx in range(len(trades)):
        volume = volume + trades[idx]['quantity']
        price = price + (trades[idx]['price'] * trades[idx]['quantity'])
    price = price / volume
    return float(price), float(volume)


class Broker(ABC):
    """ Can currently only compare two different orders on the same order book """

    def __init__(self):
        self.algo_1 = []
        self.algo_2 = []

    def place_orders(self, LOB, order_1, order_2):

        # Perform trades on orginial and copy of order book
        LOB_copy = copy.deepcopy(LOB)

        if order_1['quantity'] <= 0:
            vol_wgt_price_1, vol_1 = 0, 0
        else:
            trades_1, order_id_1 = LOB.process_order(order_1, False, False)
            vol_wgt_price_1, vol_1 = calc_volume_weighted_price_from_trades(trades_1)

        if order_2['quantity'] <= 0:
            vol_wgt_price_2, vol_2 = 0, 0
        else:
            trades_2, order_id_2 = LOB_copy.process_order(order_2, False, False)
            vol_wgt_price_2, vol_2 = calc_volume_weighted_price_from_trades(trades_2)

        self.algo_1.append(vol_wgt_price_1)
        self.algo_2.append(vol_wgt_price_2)
        return vol_wgt_price_1, vol_1, vol_wgt_price_2, vol_2

    def get_twaps(self):
        if self.algo_1:
            m_1 = np.mean(self.algo_1)
        else:
            m_1 = 0
        if self.algo_2:
            m_2 = np.mean(self.algo_2)
        else:
            m_2 = 0
        return m_1, m_2