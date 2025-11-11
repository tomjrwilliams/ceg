
import polars as pl

from .objects import *
from .values import *

def set_value_chunk_from_df(
    obj: Object,
    domain: Domain,
    df: pl.DataFrame,
    col: str,
    chunk: int,
    publish: bool = True,
):
    key_val = obj.value_key_val(domain, t = (
        ValueKeyChunkPlSeriesSQLite
        | ValueKeyChunkNumPySQLite
    ))

    # TODO: assert df has length

    if isinstance(key_val, ValueKeyChunkPlSeriesSQLite):
        v = key_val.set_from_df(
            df, col, obj.path, chunk
        )
    elif isinstance(key_val, ValueKeyChunkNumPySQLite):
        v = key_val.set_from_df(
            df, col, obj.path, chunk
        )
    else:
        raise ValueError((obj, key_val))
    
    if publish:
        e = obj.publish(ValueUpdate(obj.path), domain)
    
    return e, v

__all__ = [
    "set_value_chunk_from_df"
]