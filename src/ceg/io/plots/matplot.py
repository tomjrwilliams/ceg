from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import NamedTuple, Any
from collections import defaultdict

from frozendict import frozendict

import numpy

import matplotlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as pyplot

from ceg.core.immut import V

from ... import core

#  ------------------


class Plot(core.Plugin.Aliased):
    pass


Loc = tuple[str, int] | tuple[int, int]


class Spec(NamedTuple):
    title: str
    loc: Loc | None
    labels: frozendict[str, str]

    class Plot(NamedTuple):
        spec: Spec

        def render(self, ax: matplotlib.axes.Axes):
            return

    def key(self):
        return (
            self.title
            if self.loc is None
            else (
                self.title,
                self.loc,
            )
        )

    @classmethod
    def new(
        cls,
        title: str,
        loc: Loc = None,
        labels: frozendict[str, str] = frozendict(),  # type: ignore
    ):
        return cls(
            title=title,
            loc=loc,
            labels=labels,
        )


def render(acc: frozendict[core.Plugin.Any, Any]):
    title_specs = defaultdict(list)
    for p, v in acc.items():
        if not isinstance(p, Plot):
            continue
        assert isinstance(v, Spec.Plot), (p, v)
        spec = v.spec
        title_specs[spec.title].append(v)
    res = frozendict()
    for title, plot_specs in title_specs.items():
        plot_spec: Spec.Plot
        fig: matplotlib.figure.Figure
        if not len(plot_specs):
            continue
        elif len(plot_specs) == 1:
            (plot_spec,) = plot_specs
            fig, axes = pyplot.subplots(
                1,
                1,
            )
            axes.set_title(title)
            assert isinstance(
                axes, matplotlib.axes.Axes
            ), axes
            plot_spec.render(axes)
        else:
            raise ValueError(plot_specs)
        res = res.set(title, fig)
    return res


#  ------------------


class LineSpec_Kw(NamedTuple):
    spec: Spec
    x: core.Array.np_1D
    y: core.Array.np_1D


class LineSpec(LineSpec_Kw, Spec.Plot):

    def render(self, ax: matplotlib.axes.Axes):
        plot = self.spec
        x_label = plot.labels.get("x", "x")
        y_label = plot.labels.get("y", "y")
        ax.plot(
            x_label,
            y_label,
            "",
            data={x_label: self.x, y_label: self.y},
            # ... kwargs()
        )


#  ------------------


class Line_Kw(NamedTuple):
    scope: core.Scope.Alias | None
    spec: Spec

    # TODO: width, height, etc.


class Line(Line_Kw, Plot):
    """
    >>> from ... import fs
    >>> fs.rand.rng(seed=0, reset=True)
    >>> g = core.Graph.new()
    >>> g, ref = g.bind(None, ref=core.Ref.Col)
    >>> g, ref = g.bind(
    ...     fs.rand.gaussian.new(ref).sync(
    ...         v=core.loop.Fixed(1)
    ...     ),
    ...     ref=ref,
    ...     using=Line.new(title="rand").alias(
    ...         "y"
    ...     ),
    ... )
    >>> g, es = g.steps(core.Event(0, ref), n=10)
    >>> g, res = g.flush(es[-1])
    >>> res = render(res)
    >>> {k: type(v) for k, v in res.items()}
    {'rand': <class 'matplotlib.figure.Figure'>}
    """

    # TODO: transpose kwarg
    # eg. plot y on the x axis, assume time as y
    # as otherwise, if just plotting cols, can flip already

    @classmethod
    def new(
        cls,
        title: str,
        loc: Loc = None,
        labels: frozendict[str, str] = frozendict(),  # type: ignore
        scope: (
            core.Scope.Alias | None
        ) = core.Scope.Alias.new(),
    ):
        return cls(
            scope=scope,
            spec=Spec.new(
                title=title,
                loc=loc,
                labels=labels,
            ),
        )

    def plot_spec(
        self,
        x: core.Array.np_1D,
        y: core.Array.np_1D,
    ):
        return LineSpec(spec=self.spec, x=x, y=y)

    def flush(
        self,
        graph: core.Graph,
        event: core.Event,
        acc: frozendict[core.Plugin.Any, Any],
        scope: core.Scope.Any | None,
    ):
        assert isinstance(scope, core.Scope.Alias), (
            self,
            scope,
        )

        aliases = {
            k: ref for ref, k in scope.aliases.items()
        }
        x = aliases.get("x")
        y = aliases.get("y")

        assert y is not None, self
        assert isinstance(y, core.Ref.Col), y

        if x is not None:
            assert isinstance(x, core.Ref.Col), x
            v_x = graph.select(x, event, t=False)
            v_y = graph.select(y, event, t=False)
        else:
            v_x, v_y = graph.select(y, event, t=True)

        return acc.set(self, self.plot_spec(v_x, v_y))


#  ------------------
