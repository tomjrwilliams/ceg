
import datetime

import ib

if __name__ == "__main__":
    contr = ib.contract(
        symbol="IYJ",
        exchange="SMART",
        secType="STK",
        currency="USD",
    )
    # contr=contract(
    #     conId=12087792,
    #     underlying="EUR",
    #     symbol="EUR.USD",
    #     secType="CASH", # fx
    #     exchange="IDEALPRO",
    #     currency="USD",
    # )
    ib.get_contract(
        contr,
        end=datetime.date(2025, 1, 31),
        duration="10 D",
        bar_size="1 day",
        bar_method="MIDPOINT",
    )