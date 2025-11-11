

# TODO: you want to be able to have write-able lists that dont set off on-change detection?


# eg. append an item to a named list of universe variants at the root level
# eg. if your futures universe are ordered by asset class
# or you add a risk-off-{i}
# or risk-on-{i}


# they should not trigger recalc of variants before: where they wouldnt, i guess the enumeration generates all the variants we want to run
# the old ones wont have changed, but we'll just find some new ones


# where they're named, it wont matter anyway (as named
# where they're indexed, it wont matter so long as the position in the final list is the same for already generated variants
# else itll force a shuffle of all the items whose index changed




# two portfolios can share a parent in the hierarchy

# so they refer to the same enums

# so the signals can be implemented as trend.sig. ...

# can even have a path per asset?

# but probably not: return a vector of signals across the fixed universe?




# then here we have a vector of final positions




# enumerating over:

# universe

# signal construction, params

# eg. signal can be named, params is cartesian enum

# position unit, params

# eg. signal can be named, position is cartesian enum


# holding periods? rebal freq?