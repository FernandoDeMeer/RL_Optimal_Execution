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
    """ Currently only for placing trades and getting volume weighted execution prices """

    def place_order(self, lob, order):

        # Perform trades on order book
        if order['quantity'] <= 0:
            vol_wgt_price, vol = 0, 0
        else:
            trades, _ = lob.process_order(order, False, False)
            if trades:
                vol_wgt_price, vol = calc_volume_weighted_price_from_trades(trades)
            else:
                vol_wgt_price, vol = 0, 0

        out_dict = {'lob': lob, 'pxs': vol_wgt_price, 'qty': vol}
        return out_dict
