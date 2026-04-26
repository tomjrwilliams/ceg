
from typing import Literal, TypeAlias
import datetime as dt
from dataclasses import dataclass

from functools import lru_cache

import pytest

import polars as pl
import numpy as np

from frozendict import frozendict

import ceg

from ceg.engine import *

from ceg.objs.dev.rand import *
from ceg.objs.dev.calc.d import *

from dataclasses import fields
from typing import ParamSpec, TypeVar, Type, Callable, Any, ClassVar
P = ParamSpec("P")

class ArgBase:
    pass

O = TypeVar("O", bound=Object)


@dataclass(frozen=True)
class TGraph(Graph):
    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            Project
            | StorageConfig
            | DailyGaussian.Type
            | DailyNoise
            | DailyHAggregate.Sum
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            DailyGaussian.Type.Snap
            | DailyNoise.Snap
            | DailyHAggregate.Sum.Snap
        )

ROOT = "test_gaussian"

# TIDIL test both for sqlite and redis

def test_setup():
    from redis import Redis
    conn = Redis("localhost")
    keys = conn.keys("test/*")
    if len(keys):
        conn.delete(*keys)

@pytest.mark.order(after="test_setup")
def test_0():
    root = ROOT
    with context.graph("test_graph", t=TGraph):
        project = Project.new(root)
        project.set_storage_defaults(["config", root])
        obj = DailyGaussian.new(
            project,
            dt.date(2026, 1, 1),
            dt.date(2026, 1, 30),
            mu=0.,
            sd=1.,
        )
        context.get().with_obj(obj)
        context.get().sync()
        df = daily_pl_df(obj)
    v = df.pipe(col_as_np, obj.id)
    assert round(np.mean(v)) == 0, df
    assert round(np.std(v)) == 1, df

@pytest.mark.order(after="test_0")
def test_1():
    # NOTE: if we only issue event on snap change
    # we update the graph sync values, but dont re issue event
    root = ROOT
    with context.graph("test_graph", t=TGraph):
        project = Project.new(root)
        project.set_storage_defaults(["config", root])
        obj = DailyGaussian.new(
            project,
            dt.date(2026, 1, 1),
            dt.date(2026, 1, 30),
            mu=0.,
            sd=1.,
        )
        noise = DailyNoise.new(
            obj,
            mu=0.,
            sd=0.1,
        )
        noise2 = DailyNoise.new(
            obj,
            mu=0.,
            sd=0.1,
            suffix="2"
        )
        agg = DailyHAggregate.sum(
            project,
            params = frozendict({
                noise.id: noise.id,
                noise2.id: noise2.id,
            }),
            kwargs=frozendict({
                "fire_every_time": True,
            }),
            xid=0,
        )
        context.get().with_obj(obj)
        context.get().with_obj(noise)
        context.get().with_obj(noise2)
        context.get().with_obj(agg)
        context.get().sync(run=0)
        df = daily_pl_df([obj, noise, noise2, agg])

        obj_e = obj.get_event()
        agg_e_0 = agg.get_event(0)
        agg_e_1 = agg.get_event()

    v_obj = df.pipe(col_as_np, obj.id)
    v_noise2 = df.pipe(col_as_np, noise2.id)
    v_noise = df.pipe(col_as_np, noise.id)
    v_agg = df.pipe(col_as_np, agg.id)
    v_delta = (v_noise - v_obj) * 10
    assert round(np.mean(v_delta)) == 0, df
    assert round(np.std(v_delta)) == 1, df
    assert np.allclose(v_agg, v_noise + v_noise2), df

    assert agg_e_1.i == 2, agg_e_1
    assert isinstance(agg_e_0, ObjectNew), agg_e_0
    assert isinstance(agg_e_1, ValueUpdate), agg_e_1

    context.drop("test_graph")

    with context.graph("test_graph", t=TGraph):
        project = Project.new(root)
        project.set_storage_defaults(["config", root])

        context.get().with_obj(obj)
        context.get().with_obj(noise)
        context.get().with_obj(agg)

        context.get().sync(run=1)

        assert obj_e.i == obj.get_event().i - 1, (
            obj_e, obj.get_event()
        )

        agg_e = agg.get_event()
        assert agg_e.i == agg_e_1.i, (
            agg_e, agg_e_1
        )

        context.get().with_obj(noise2)
        context.get().sync(run=2)

        # NOTE: we catch up to both sync at the same time
        # so +2 is technically correct

        agg_e = agg.get_event()
        assert agg_e.i == agg_e_1.i + 2, (
            agg_e, agg_e_1
        )


@pytest.mark.order(after="test_1")
def test_2():
    root = ROOT

    c1 = Collection("c1")
    c2 = Collection("c2")

    o1: list[Bound] = [
        DailyGaussian.bind(
            "g1",
            "o:project",
            dt.date(2026, 1, 1),
            dt.date(2026, 1, 30),
            mu=0.,
            sd=1.,
            suffix="g1"
        ),
        Bound(
            DailyNoise.new,
            "n1",
            params = bind_params(
                bound("g1"),
                mu=0.,
                sd=0.1,
                suffix="n1"
            )
        ),
        Bound(
            DailyNoise.new,
            "n2",
            params=bind_params(
                bound("g1"),
                mu=0.,
                sd=0.1,
                suffix="n2"
            )
        )
    ]
    o2: list[Bound] = [
        DailyGaussian.bind(
            "g2",
            "o:project",
            dt.date(2026, 1, 1),
            dt.date(2026, 1, 30),
            mu=0.,
            sd=1.,
            suffix="g2"
        ),
        Bound(
            DailyNoise.new,
            "n3",
            params = bind_params(
                bound("g2"),
                mu=0.,
                sd=0.1,
                suffix="n3"
            )
        ),
        Bound(
            DailyNoise.new,
            "n4",
            params=bind_params(
                bound("g2"),
                mu=0.,
                sd=0.1,
                suffix="n4"
            )
        ),
        Bound(
            DailyHAggregate.sum,
            "sum",
            params=bind_params(
                "o:project",
                params = (
                    bound("n3"),
                    bound("n4")
                ),
                kwargs=frozendict({
                    "fire_every_time": True,
                }),
                xid=1,
            )
        ).unpack("params")
    ]

    with context.graph("test_graph", t=TGraph):
        project = Project.new(root)
        project.set_storage_defaults(["config", root])

        c1 = c1.with_obj(project, "project")
        for o in o1:
            c1 = o.bind(c1)

        c2 = c2.with_obj(project, "project")
        for o in o2:
            c2 = o.bind(c2)

        c = Collection.merge(c1.set_prefix(), c2.set_prefix())

        c = Bound(
            DailyHAggregate.sum,
            "sum",
            params=bind_params(
                "o:project",
                params = (
                    bound("c1/n1"),
                    bound("c1/n2"),
                    bound("c2/n3"),
                    bound("c2/n4"),
                ),
                kwargs=frozendict({
                    "fire_every_time": True,
                }),
                xid=2,
            )
        ).unpack("params").bind(c.with_obj(project, "project"))

        context.get().sync()

        # TODO: get back from collections, with filtering
        objs = {
            k: assert_is(o, Daily)
            for k, o in c.objects.items()
            if "project" not in k
        }
        df = daily_pl_df(list(objs.values())).rename({
            o.id: k for k, o in objs.items()
        })

    assert np.allclose(
        df.pipe(col_as_np, "sum"),
        df.pipe(col_as_np, "c1/n1")
        + df.pipe(col_as_np, "c1/n2")
        + df.pipe(col_as_np, "c2/sum")
    ), df


@pytest.mark.order(after="test_2")
def test_3():
    root = ROOT

# TODO: buffer

# test works correctly before after write boundary
# load and reload at each step, only after, etc.


@pytest.mark.order(after="test_3")
def test_4():
    root = ROOT

# TODO: non-self snap return

# update snap for other than self - check handled still correct for self (as no snap update returned)


@pytest.mark.order(after="test_4")
def test_5():
    root = ROOT

# TODO: flush 

# flush down from a given project / root (all?) data (and events etc.?)

# version tag causes downstream value-update cascade? means listening for deps to the project node?


@pytest.mark.order(after="test_5")
def test_6():
    root = ROOT

# TODO: multi proc

# no batching

# batching (those that dont return / snap one another can be batched together)

# multi proc respects snap order / batching


@pytest.mark.order(after="test_6")
def test_7():
    root = ROOT
    
# TODO: sqlite tests?
# TODO: redis benchmark relative performance?