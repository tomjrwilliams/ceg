
# import ceg

from redis import Redis

conn = Redis(
    "localhost",
    decode_responses=False
)

conn.flushall()