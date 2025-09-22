# relational query API to add meta data to the keys

# and join to fixed universes eg. tickers, etc.


# so need to define Categoricals (universes that get ids)

# and then join tables between them


# and both one another and key prefixes


# and then ways of traversing / building up queries over the keys


# can eg. i guess add meta data, snap (?), other tag paths as well


# ideally the tags themselves are applied *as* relational joins

# where the tag could be to a sepcific ticker, but we then pre load a join from the specific ticker to the chain (ES25 -> ES etc.), asset class, etc.


# TODO: the tags you use for index path nodes can be richly interpreted as eg. a categorical (can even use the keyval API to store and get back an ID)

# and then whether it's a column or a join is decided at def time

# i guess the diff is one to many vs many to one, many to many, etc.

# eg. asset class is more a foreign key on the table

# individual constituents for a single, if many, will likely be a join

# TODO: so that makes sense if the tag is plural (and each have ID), then you use a join table not a foreign key

# TODO: teh last q is then how to define other static mappings between tags

# ie. you might tag with an already richly defined data object?

# ie. the ticker in the case above, that might be inside a list, itself might contain other join / foreign key / tag relations?

# does that then need to be under the same versioning as the index paths?
