

# arrival process for orders
# sometimes state change on process parameter (s)

# initially fixed:

# profit per sale
# cost per inventory held

# later can allow cost of production to vary
# and price of sale
# and inventory holding

# sale starts off as 100% probability given order, if inventory
# later can make probabilistic 

# agent chooses production
# starts off immediate arrival (?)
# later add lag (can then make probabilistic)

# each time:

# agent finds out sales that period
# calculates profit in that period

# then two options
# rl or parametric optimisation

# parametric, likely two steps

# eg. fit agent estimate of model parameters (eg. arrival process rate)
# and then given arrival estimate
# and current inventory, unit profit, holding cost
# pick production so as to maximise future discounted reward from here


# rl you need a single parametric function from current state and new data to production
# there its also possibly easier to separate process param estimation
# and production given arrival param

# ie. start with just a linear model over the unit costs, inventory, and arrival rate
# can later make more complicated as intuition develops

# that you then fit RL style based on the observed reward each time
# but then have to fit the params of the linear model by running many times



# vs the parametric approach where the maximisation is determined each stype by the functional form, so there's not the same fitting
# rather finding the right functional form for your future expected reward that you can optimise it directly (given arrival estimate)





# what are the outputs?

# arrival process param
# orders time series

# agents arrival process param estimate

# agent inventory time series (including newly received production)
# agent sales times series
# agent profit time series


# agent new production decision time series
# agent maximised estimated total future profit from here, time series




# exactly the same graphs would also describe the model with lag between production decision and receipt
# and with varying profit per sale / cost per unit / inventory cost


# just the complexity of the params might change, eg. if we move to a markov model with state estimates

# even a two variable model where agents optimise over both price and production
# where there you just need another model for agents that gives a sale probability given an arrival and a price