class Alloc:
    # str - or identifier? probably identifier (in key val)
    asset: str
    scale: str
    signal: str
    rebal: str  # used for roll schedule?
    quantum: float  # what to multiply by scale and signal
    # both value and units?

    # later on this might be some kind of constrained equaton and not just a single value?
    # that the scale and signal then multiply up or down?

    def exposes(self):
        # the list of fields this alloc exposes
        # pnl
        # where these should be a name and unit
        # eg. pnl in dollars
        # rx in pct on capital, on margin, on delta, etc.

        # include frequency?

        # when this alloc is used as an asset, these terms can be in the scale, signal, value for the container alloc
        return


# metrics:
# returned from exposes

# ideally we have fixed ways of mapping between certain kinds, eg. between currencies for pnl?


class Strategy:
    name: str  # identifier path rather than string, really, but there is then a unique string key for the path
    allocs: list[Alloc]

    def exposes(self):
        # intersection of children?

        # need rules for how metrics combine
        # pnl sums, but drawdown doesnt
        return


# recursive structure: we can build strategies out of other strategies
# the metrics exposes by a strategy can be used in the scale / signal etc. of a parent strategy, allocating ot it

# need ot be careful on timing: eg. pnl is onl avaialble t+1 etc. on an alloc t-1, so metric time alignemnt matters


# NOTE: sounds like we're getting to a object based layer on top of the calc graph, where we build up the calc graph given the metrics etc. above

# binding eg. nodes to particular fields on the objects for each alloc (scale,signal) (where that node might have several children for a complex calc)
# up to the startegy fields, etc. (and on)

# then run the calc graph

# then for each object, iterate over the node fields and save them down to the database by ID


# or when we bind, we do so with the relevant tag information, so that we can persist as the simulation proceeds


# TODO: the ui would be

# a table for the strategy

# with id path

# and then each alloc is a sub table with rows for

# the asset id path

# scale expression

# signal expression

# freq expression

# quantum expression (could just be a value and unit)


# buttons to edit existing, copy and create new, add / remove alloc etc.


# TODO: so the grpah engine needs a way of ingesting arbitrary path value time series

# and use that as the generic way to get eg. external data in

# we have an ingress that converts to the generic id path, field form

# and then read that in

#  --

# which we then use as the break-point between data-generation / job boundaries

# the serialisation points, where we write the graph back to the persistence with an id path

# so the next job can just read the id path and not re-calc


# TODO: it sounds like each node needs a way of describing what it returns

# to use as the metric signature

# which might have to be dynamic, ie. a function not static - eg. for if it's a reduction to a different frequency

# then we can say given the input signatures, ie. daily signautres

# and the arguments

# that it returns a given frequence (say monthly)


# these sound then like you can include as tags?

# where that seems like a second wrapper layer, on a given function call in a given context, to identify it

# as part of identifying it with a given tag-path


# so as well as the tag path, we need to store the signature of the identifier

# then we either read in with a signature, eg. daily (and units?), none treatment (?), from the identifier store

# or it should be a dynamically created node, that should be able to calc its signature given those of inputs (other dynamically created, or identifier store that should have it stored)


#
