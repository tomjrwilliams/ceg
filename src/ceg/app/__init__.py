from . import academic
from . import dev

# app has the same dev (projects) and prod distinction as objs

# as app is mostly just ui config onto the data produced in objs


# same idea in principle of dev taking over and adapting the prod pages, referring to the dev (per project) data instead



# though, there is no reason prod cant have uis that can query dev data: the distinction is rather what should always work / be considered the 'main', official, user interfaces

# rather than necessarily what data is input - prod should always work, even if its querying over dev data (so be fairly generic)

# dev might only work for the particular project match