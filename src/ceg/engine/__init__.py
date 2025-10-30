

# snap: current state of object, summary values and an id (?), updated in place

# value: the rest of the object, maybe chunked kv for time series, updated in place

# object: uniquely identified by path, parent object for traversal (object s?)

# root object is environment var registry, dev, prod etc.

# event: change in snap (?), external data change, change in version / tags

# tags: fields on objects, used to identify for a given path (so a change in function for an object is only picked up if eg. has a new version tag)

# domains: control flow boundary between update runs, iterable and buildable from one other





# single event queue: objects specify what they subscribe to, keep track of where they are
# but, if new subscription added, need to know to re-traverse (looking for that subscription)
# hence, need to track per subscription target - say, object
# in which case it's rather each object tracks the last status of its subscribed objects, starting at zero, handling until current
# ah we dont have the history? so we cant rebuild iteratively anyway, just jump to last
# in which case it doesnt matter the previous event history of the object
# rather we can take the subscription itself as an event and use that to trigger whatever needs doing?


# values: aside from the exact implementation (eg. which backend), indicate which kind (eg. time series), and meta information eg. index, if gappy or not (eg. calendar treatment)
# then different kinds of value have different sub events that can be fired
# in fact are fired by the outer implementation of shared interface for updating (eg. append / prepend / update methods for time series)
# so we need both concrete implementations for them in apis, and the abstract outer implementation in terms of the event framework



# on event doesnt need to imply snap change (value update), eg. if no op
# what about all the objects that dont subscribe to a new event id? they dont need to be all bumped

# rather its just to check relative to what we think we've processed, only the subs for an object need to check if their event id is already past the one being checked
# so objects might sit at different stages in the global queue
# and the global queue config needs to sit at the root object so everything can react if it changes


# default to implying parent implies sub so no separate object required?
# interesting to instead represent both as kinds of relationship
# which the object itself creates - return on domain init, look in db if there, check status
# parenthood can thus be turned off, if not found in the subscriptions method return
# if turned off, default for children? it's step by step down the parent tree, so effectively yes, the child has to re-subscribe to the grandparent


# call them dependencies, not subscriptions as parents are stored but not sub with field for if parent, or not
# and if active or not
# ie. we store object parents beyond just the path element, so they can be more easily queried without having to use string operations
# also if one no longer appears, deactivate it (and an event presumably - though less there to do?)


# i think you probably want to separate different kinds of value update: append / prepend only, from genuine value overwrite, in the events, so you can decide downstream how to act


# event handling is thus a function of the object string, and the event itself
# presumably at the subscribes method call, you create everything required to handle that subscription
# but the handlers and the subscription implementation is not versioned semantically: you have to update the version manually in the code, and handle your own update (ie. regen data) accordingly


# the domain just checks if new or not, but retains the callback / handlers for processing


# domain.sync: call on sync for external event checks, then check every object with event state < current event queue
# iterate through the events since the earliest, looking for matches?
# much more efficient to store the global event counter in the domain itself, so it can simply check from there against the objects in scope

# where we therefore allow objects to be updated by multiple domains, and we'd skip as their event would be after the current domain event

# domains thus also have some persisted state, but they're orthogonal to the value and dependency apparatus
# very simple: methods to acc objects, subscriptions (check for events), handlers, on_sync
# and then sync, outlined above