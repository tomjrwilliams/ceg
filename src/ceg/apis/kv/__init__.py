# backends determine permissive i types, what goes in
# I is fixed per backend (store)

# front end determines o types, map to i (both key and value)
# o ideally is broad as possible: np, polars, etc.

# o is business logic, needs to implement codign for whatever output (business) types you have, given the input stores you intend to use

from . import core
from . import fs
from . import sql

from .core import (
    # _,
    T,
    ID,
    KeyI,
    ValI,
    KeyO,
    ValO,
    ValT,
    not_none,
    #
    KeyValConfig,
    KeyValStore,
    KeyValIO,
    KeyVal,
    #
    KeyValStrKeyI,
    KeyValBytesValI,
    KeyValStrKeyIBytesValI,
    #
    MemoryKeyVal,
    ExampleKeyVal,
    #
    HEAD,
    BODY,
    TAIL,
    #
    KeyChunkConfig,
    KeyChunkStore,
    KeyChunkIO,
    KeyChunk,
    #
    MemoryKeyChunk,
    ExampleKeyChunk,
    #
    QueueConfig,
    QueueStore,
    QueueIO,
    Queue,
    QueueTimestampAsFloat,
)
from .sql.lite import (
    SQLiteKeyValConfig,
    SQLiteKeyVal,
    SQLiteKeyChunkConfig,
    SQLiteKeyChunk,
    SQLiteQueue,
    SQLiteQueueConfig,
)