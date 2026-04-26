

import ceg

df = ceg.apis.data.fred.load_time_series(
    "T10YIE",
    cache="./__local__/data/fred"
)

print(df)