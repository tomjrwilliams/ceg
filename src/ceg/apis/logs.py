
import os
import logging

TEST_LOGS = os.environ.get("TEST_LOGS", None)

def prepare_logger(__file__: str):
    logger = logging.Logger(__file__)
    
    if TEST_LOGS is not None:
        log_handler = logging.FileHandler(TEST_LOGS)
        log_handler.setLevel(logging.DEBUG)
        logger.addHandler(log_handler)

    def log_kw(
        level = logging.DEBUG,
        prefix: str | None = None, 
        **kwargs
    ):
        if prefix is None:
            return logger.log(level, repr(kwargs))
        logger.log(level, prefix)
        for k, v in kwargs.items():
            if hasattr(v, "log_repr"):
                s = v.log_repr()
            else:
                s = repr(v)
            logger.log(level, f"\t{k}={s}")
        return
    
    return logger, log_kw