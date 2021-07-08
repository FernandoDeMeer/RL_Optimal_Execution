from env.orderbook import OrderBook
import json
from decimal import Decimal


def split_book_to_orders(current_book):
    """ Splits existing order book data into individual bid and ask orders """

    # pre-allocate
    bid_orders = []
    ask_orders = []

    # get meta info
    time_stamp = current_book["T"]
    trade_id = 0

    # extract bid orders
    for bid in current_book["b"]:
        bid_order = {'type' : 'limit',
                     'time_stamp': time_stamp,
                     'side' : 'bid',
                     'quantity' : Decimal(bid[1]),
                     'price' : Decimal(bid[0]),
                     'trade_id' : trade_id}
        trade_id += 1
        bid_orders.append(bid_order)

    # extract ask orders
    for ask in current_book["a"]:
        ask_order = {'type' : 'limit',
                     'timestamp': time_stamp,
                     'side' : 'ask',
                     'quantity' : Decimal(ask[1]),
                     'price' : Decimal(ask[0]),
                     'trade_id' : trade_id}
        trade_id += 1
        ask_orders.append(ask_order)
    all_orders = bid_orders + ask_orders

    return bid_orders, ask_orders, all_orders


if __name__ == '__main__':

    # some input
    n_snapshots = 1

    # instantiate order book
    order_book = OrderBook()

    # open file
    file = open("./data/book_depth_socket_btcusdt_2021_06_21.txt", "r+")

    """ 
    transfer orders to order book
    
        NOTE: 
            * this is currently just correct for a single snapshot.
            * if we were to place a hypothetical trade in this snapshot of an order book, we'd have to 
            factor in the dynamics of this trade for simulation of prices
    
    """
    # transfer orders to order book (NOTE: this is currently just correct for a single snapshot)
    line_counter = 1
    for line in file:
        if not line_counter > n_snapshots:
            current = json.loads(line)
            _, _, all_orders = split_book_to_orders(current)
            for order in all_orders:
                trades, order_id = order_book.process_order(order, False, False)
                line_counter += 1
        else:
            break
    file.close()

    # print current book
    print(order_book)



