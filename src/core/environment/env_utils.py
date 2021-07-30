from decimal import Decimal
from src.core.environment.orderbook import OrderBook


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


def raw_to_order_book(current_book, depth):
    order_book = OrderBook()
    bid_orders, ask_orders, _ = split_book_to_orders(current_book)

    counter = 1
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
    return order_book