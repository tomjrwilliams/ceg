


# NOTE: where loading a pl series, we can return both the values and a boolean null mask (separate to nan values)
# can then decide whether to emit none or not (etc)

# NOTE: whereas if a numpy result then nan=null, so have to emit (unless a known not-nan node, in which case assume none and filter)



# generic methods for loading object series by name
# and type: eg. dense daily time series, not dense

# has none vs not



# eg. all the pricing data
# and intermediate results, prepared by other domains




# then its up to the graph itself to expose for storage the results (in the histories)
# in a way thats easy to access



# each graph becomes quite small, defined between the caching boundaries of the domains