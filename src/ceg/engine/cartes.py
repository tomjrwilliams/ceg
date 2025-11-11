

# utils for generating universes from as succint a representation as possible

# where each item in a cartesian product result (poss filtered) is given a unique index

# either a function of the result values, or int


# and then values for unpacking these at run time
# as then the domain auto-rebuilds on run, compared against what's currently stored at each index




# so the objects themselves are much more informative than the names, becuase they end up containing all the final information
# albeit - and need utils for this - possibly built up at several layers of depth?





# ie. {universe}{methodology}{parameters}
# as a path over several enumerated products


# pfo.trend.0.2.4
# pfo.crisis. ... 









# so the object index path is used to identify a config iteration in a parent object

# possibly multiple layers deep?


# ie. {universe}{methodology}{parameters}

# could each be themselves separate cartesian products, that we generate paths for each combination through


# and the objects can hold the concrete values, even if the parents are lazy - in fact they have to, as thats how we know what's changed
# even though their identifier is just like:
# pfo.trend.0.2.4
# pfo.crisis. ... 

# and then we just iter the whole range on load



# optionally allow for naming particular cartesian product results, as long as they're unique

# 