from __future__ import annotations
import os
import logging
from pathlib import Path
from functools import partial, wraps
from typing import (
    Unpack,
    TypedDict,
    Any,
    Concatenate,
    Callable,
    TypeVar,
    Literal,
    Protocol,
    Type,
    TypedDict,
    Any,
    ParamSpec,
    overload,
    cast,
    runtime_checkable,
)

from inspect import Parameter

EMPTY = Parameter.empty()
Empty = Parameter.empty

from .validation import assert_is, EMPTY, is_empty

TEST_LOGS = os.environ.get("TEST_LOGS", None)


class Handlers:
    @staticmethod
    def file(
        filename: str | Path,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
        level: logging._Level | None = None,
        formatter: Literal[True]
        | logging.Formatter
        | None = None,
        # TODO: formatter
    ):
        if formatter == True:
            formatter = Formatter()
        handler = logging.FileHandler(
            filename=filename,
            mode=mode,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )
        if level is not None:
            handler.setLevel(level)
        if formatter is not None:
            handler.setFormatter(formatter)
        return handler

    @staticmethod
    def stream(
        stream: Any | None = None,
        level: logging._Level | None = None,
        formatter: Literal[True]
        | logging.Formatter
        | None = None,
    ):
        if formatter == True:
            formatter = Formatter()
        handler = logging.StreamHandler(stream=stream)
        if level is not None:
            handler.setLevel(level)
        if formatter is not None:
            handler.setFormatter(formatter)
        return handler


class Formatter(logging.Formatter):
    width: int = 32
    datefmt: str = "%H:%M:%S"

    def format(self, record: logging.LogRecord):

        cpath = "%s:%s:%s" % (
            record.module,
            record.funcName,
            record.lineno,
        )
        cpath = cpath[-self.width :].ljust(self.width)

        prefix = "%-7s %s %s" % (
            record.levelname,
            self.formatTime(record, self.datefmt),
            cpath,
        )

        message = record.getMessage()

        record_dict = record.__dict__
        log_keys = set(record_dict.keys()).difference(
            record_keys
        )
        if (len(log_keys) + (1 if len(message) else 0)) > 1:
            s = "---\n"
        else:
            s = ""

        s += f"{prefix} : {message}" if len(message) else ""

        for k in log_keys:
            v = record_dict[k]
            if isinstance(v, HasLogRepr):
                v_s = v.log_repr()
            else:
                v_s = repr(v)
            s += (
                "\n" if len(s) and not s.endswith("\n") else ""
            ) + f"{prefix} : {k}={v_s}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(
                    record.exc_info
                )

        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text

        # if s.endswith("\n"):
        #     s.removesuffix("\n")
        return s.rstrip()


levels = logging.getLevelNamesMapping()
record_keys = set(
    logging.LogRecord(
        "", 0, "", 0, "message", None, None
    ).__dict__.keys()
)
record_keys.add("message")


def logger(
    __file__: str,
    files: dict | list[dict] | None = None,
    streams: dict | list[dict] | None = None,
    level: logging._Level | None = None,
    formatter: Literal[True]
    | logging.Formatter
    | None = None,
    when: Callable[[dict], bool] | None = None,
):

    if formatter == True:
        formatter = Formatter()

    logger = logging.Logger(__file__)

    if level is not None:
        logger.setLevel(level)

    files = (
        []
        if files is None
        else [files]
        if isinstance(files, dict)
        else files
    )
    streams = (
        []
        if streams is None
        else [streams]
        if isinstance(streams, dict)
        else streams
    )

    if TEST_LOGS is not None:
        files.append(
            dict(filename=TEST_LOGS, level=logging.DEBUG)
        )

    for kws in files:
        # TODO: level likewise?
        kws["formatter"] = kws.get("formatter", formatter)
        logger.addHandler(Handlers.file(**kws))

    for kws in streams:
        # TODO: level likewise?
        kws["formatter"] = kws.get("formatter", formatter)
        logger.addHandler(Handlers.stream(**kws))

    @wrap_log(logger.log)
    def log(
        level: logging._Level = level or logging.DEBUG,
        value: Empty | Any = EMPTY,
        **kwargs,
    ):
        if isinstance(level, str):
            level = levels[level]
        if is_empty(value):
            value = ""

        # TODO: return a new func type annot
        # with eg. tags etc.

        def f(**extra: Any):
            if when is None or when(extra):
                _ = extra.pop("when", None)
                logger.log(
                    level,
                    value,
                    extra=extra,
                    stacklevel=2,
                    **kwargs,
                )

        return f

    return logger, log


type Level = logging._Level

P = ParamSpec("P")
T = TypeVar("T")


def wrap_log(
    f: Callable[Concatenate[Any, Any, P], T],
) -> Callable[
    [Callable],
    Callable[Concatenate[int | str, P], Callable[..., T]],
]:
    return wraps(f)


@runtime_checkable
class HasLogRepr(Protocol):
    def log_repr(self) -> str: ...


__all__ = [
    "Handlers",
    "logger",
]
