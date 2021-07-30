from src.core.environment.orderbook import OrderBook
from decimal import Decimal


def split_book_to_orders(current_book, time, depth):
    """ Splits existing order book data into individual bid and ask orders """

    # pre-allocate
    bid_orders = []
    ask_orders = []

    # get meta info
    trade_id = 0

    # extract ask orders
    for ask_idx in range(0, min(depth, len(current_book[0]))):
        ask_order = {'type' : 'limit',
                     'timestamp': time,
                     'side' : 'ask',
                     'quantity' : Decimal(str(current_book[1][ask_idx])),
                     'price' : Decimal(str(current_book[0][ask_idx])),
                     'trade_id' : trade_id}
        trade_id += 1
        ask_orders.append(ask_order)

    # extract bid orders
    for bid_idx in range(0, min(depth, len(current_book[2]))):
        bid_order = {'type' : 'limit',
                     'time_stamp': time,
                     'side' : 'bid',
                     'quantity' : Decimal(str(current_book[3][bid_idx])),
                     'price' : Decimal(str(current_book[2][bid_idx])),
                     'trade_id' : trade_id}
        trade_id += 1
        bid_orders.append(bid_order)

    all_orders = bid_orders + ask_orders
    return bid_orders, ask_orders, all_orders


def raw_to_order_book(current_book, time, depth):
    """ Convert the raw LOB data into an OrderBook object """

    # transfer numerical data to orders
    _, _, all_orders = split_book_to_orders(current_book, time, depth)
    order_book = OrderBook()
    for orders in all_orders:
        _, _ = order_book.process_order(orders, False, False)

    # place orders in OrderBook class
    """
    counter = 1
    order_book = OrderBook()
    for bids in bid_orders:
        _, _ = order_book.process_order(bids, False, False)
        counter += 1
        if counter > depth:
            break

    counter = 1
    for asks in ask_orders:
        _, _ = order_book.process_order(asks, False, False)
        counter += 1
        if counter > depth:
            break
    """
    # check that the order book has been generated correctly and nothing has been cancelled...
    return order_book