
from . import db
from . import raw

# themes (systematic):

# - past returns predictive of forward, between assets
# past returns -> past factor -> forward factor -> forward returns

# - past cum delta to factor expected, predictive of forward
# past returns -> past factor -> past expected returns -> forward returns

# themes (discretionary):

# - forward return estimate -> other assets
# forward estimate -> forward factor -> forward other assets

# assuming get macro data points as well (inflation, gdp)

# eg. i think oil will go 5% in 3m, growth will stall out, range of scenarios for other macro assets?

# so eg. table of mean, std dev
# possibly mixture model (mean std dev)

# possibly conditional prob of mixture states across assets / factors

# can then simulate (?) future paths
# optimise over portfolio choice given future paths

# and or backtest specific portfolios, under those paths

# so both given factor / asset distribution mixtures and correlations (on distribution states)
# and a portfolio

# can generate a probabilistic estimate of forward portfolio performance