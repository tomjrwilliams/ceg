from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from .. import apis
from ..apis import t, kv, assert_is

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *
from .storage import *


@dataclass(frozen=True)
class DependencyStatus:
    past: bool
    hit: bool
    event: Event
    dep: Dependency


# q = self.root.event_key_val(self)
# history = q.head_with_i(n=-1)


def dependency_status(
    obj: Object,
    history: list[Event],
):
    # snap = assert_is(self.root.get_snap(self))
    # last = snap.event

    # targets = {}
    # depends = obj.depends(self)

    # if not len(depends):
    #     return {}

    # for dep in depends:
    #     if not dep.active:
    #         continue
    #     assert dep.obj == obj.id, (dep, obj)
    #     target = dep.depends
    #     if target not in targets:
    #         targets[target] = {}
    #     events = (
    #         [None] if dep.events is None else dep.events
    #     )
    #     for e in events:
    #         if e not in targets[target]:
    #             targets[target][e] = []
    #         targets[target][e].append(dep)

    # if not len(targets):
    #     return {}

    # result = {target: {
    #     t: [] for t, _ in type_deps.items()
    # } for target, type_deps in targets.items()}

    # for e in history:
    #     if not e.obj in targets:
    #         continue
    #     event_deps = targets[e.obj]
    #     for e_type, deps in event_deps.items():
    #         if e_type not in result[target]:
    #             result[target][e_type] = []
    #         result[e.obj][e_type].extend([
    #             DependencyStatus(
    #                 assert_is(e.i)
    #                 <= assert_is(last.i),
    #                 e_type is None or e.type == e_type,
    #                 e,
    #                 dep
    #             )
    #             for dep in deps
    #         ])

    # return result
    # TODO: implement
    raise ValueError(obj)


# TODO:
# find objects that dont have snap
# for each, get deps (or those of what creates - eg. rx has no dep, is created by bt?) pf pos likewise
# then back track all deps that arent self ObjectChange
# until we get to one that does have a snap?
# or identify a circularity?
# or identify something not even in the domain?
# or, what else might be wrong?

# ---


def status_report(examples: list[str]):

    # e_q = domain.root.event_key_val()
    # history = e_q.head_with_i(n=-1)

    # for obj in [
    #     context.get().get(s, Object)
    #     for s in examples
    # ]:
    #     deps = dependency_status(
    #         domain,
    #         obj,
    #         history,
    #     )
    #     print("--")
    #     print("obj:", obj.id)
    #     for target, events in deps.items():
    #         print("dep:", target, {
    #             t: len(es)
    #             for t, es in events.items()
    #         })
    # TODO: implement
    raise ValueError()


# if name == "forever":
#     objs = [
#         "p006-forever-6-bt-uv-000",
#         "p006-forever-6",
#     ]
#     engine.queries.status_report(objs)

__all__ = [
    "DependencyStatus",
    "dependency_status",
]
