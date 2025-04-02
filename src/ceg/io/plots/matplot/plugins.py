from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import NamedTuple, Any, Optional, Type
from collections import defaultdict

from frozendict import frozendict

import numpy

import matplotlib
import matplotlib.lines
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as pyplot

from . import core
from .... import core as ceg

from .core import Grid_Key

#  ------------------

t_slice = slice

def continuous_1d_y(
    mark: Continuous_1D,
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
    t: str | None = None,
    c: str | None = None,
):
    x = x2 = c = None
    tt, y = graph.select(ref, at=event, t=True)
    if isinstance(slice, (int, t_slice)):
        y = y[slice]
        tt = tt[slice]
    elif isinstance(ref, ceg.Ref.Col):
        pass
    elif isinstance(ref, ceg.Ref.Col1D):
        y = y[-1]
        tt = None
    if t is not None and t == mark.x:
        x = tt
    else:
        assert x is None, x
    if t is not None and t == mark.x2:
        x2 = tt
    else:
        assert x2 is None, x2

    # TODO: equals any of the mark fields
    # if c == "y":
    #     c = y
    # elif c == "t":
    #     c = t
    # else:
    #     assert c is None
    return y, x, x2, c

def continuous_1d_x(
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
):
    x = graph.select(ref, at=event, t=False)
    if isinstance(slice, (int, t_slice)):
        x = x[slice]
    elif isinstance(ref, ceg.Ref.Col):
        raise ValueError(ref)
    elif isinstance(ref, ceg.Ref.Col1D):
        x = x[-1]
    return x

def continuous_1d_c(
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
):
    x = graph.select(ref, at=event, t=False)
    if isinstance(slice, (int, t_slice)):
        x = x[slice]
    elif isinstance(ref, ceg.Ref.Col):
        raise ValueError(ref)
    elif isinstance(ref, ceg.Ref.Col1D):
        x = x[-1]
    return x

def continuous_1d_kwargs(
    mark: Continuous_1D,
    aliases: frozendict[
        str, tuple[ceg.Ref.Any, frozendict]
    ],
    graph: ceg.Graph,
    event: ceg.Event
):
    x = None
    y = None
    x2 = None
    y2 = None
    c = None

    if mark.y is not None and mark.y in aliases:
        ref, kwargs = aliases[mark.y]
        y, x, x2, c = continuous_1d_y(
            mark,
            ref, # type: ignore
            graph, event, **kwargs
        )

    if mark.y2 is not None and mark.y2 in aliases:
        ref, kwargs = aliases[mark.y2]
        y2, x, x2, c = continuous_1d_y(
            mark,
            ref, # type: ignore
            graph, event, **kwargs
        )

    assert y is not None or y2 is not None, aliases

    if x is None and mark.x is not None and mark.x in aliases:
        ref, kwargs = aliases[mark.x]
        x = continuous_1d_x(
            ref,  # type: ignore
            graph, event, **kwargs)

    if x is None and mark.x2 is not None and mark.x2 in aliases:
        ref, kwargs = aliases[mark.x2]
        x2 = continuous_1d_x(
            ref,  # type: ignore
            graph, event, **kwargs)

    if c is None and mark.c is not None and mark.c in aliases:
        ref, kwargs = aliases[mark.c]
        c = continuous_1d_c(
            ref, # type: ignore
            graph,
            event,
            # slice from kwargs?
        )

    return dict(
        x=x,
        y=y,
        x2=x2,
        y2=y2,
        c=c,
        # x_label, etc.
    )

class Continuous_1D_Kw(NamedTuple):
    scope: ceg.Aliases | None
    grid: Grid_Key
    figure: Optional[str]
    axis: str
    x: str | None = None
    y: str | None = None
    x2: str | None = None
    y2: str | None = None
    c: str | None = None
    slice: int | None = None # or slice
    colors: Optional[core.Color | core.Colors] = None

class Continuous_1D(Continuous_1D_Kw, ceg.Plugin.Aliased):

    @classmethod
    def new(
        cls,
        grid: Grid_Key, # needs all plots already added
        axis: str,
        scope: ceg.Aliases | None = None,
        figure: Optional[str] = None,
        # x / y etc. refer to aliases
        x: Optional[str] = None,
        y: Optional[str] = None, # label for series
        x2: Optional[str] = None,
        y2: Optional[str] = None,
        c: Optional[str] = None,
        colors: Optional[core.Color | core.Colors] = None,
    ):
        return cls(
            scope=scope,
            grid=grid,
            axis=axis,
            figure=figure,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            c=c,
            colors=colors,
        )

    def plot(self) -> Type[core.Continuous_1D]:
        raise ValueError()

    def flush(
        self,
        graph: ceg.Graph,
        event: ceg.Event,
        state: ceg.State,
        scope: ceg.Aliases | None,
    ):
        assert isinstance(scope, ceg.Aliases), (
            self,
            scope,
        )

        aliases = frozendict({
            key: (ref, kwargs)
            for (ref, key), kwargs 
            in scope.aliases.items()
        })
        kwargs = continuous_1d_kwargs(
            self,
            aliases, # type: ignore
            graph, event
        )

        grid: core.Grid = self.grid.get(graph)

        grid = grid.with_chart(
            # TODO: figure etc.
            getattr(core.fig.axis, self.axis),
            self.plot().new(
                **kwargs,
                colors=self.colors,
                # TODO: labels
                # TODO: self.kwargs
            ),
        )
        
        return state.set(self.grid, grid)


#  ------------------

class Line(Continuous_1D):
    """
    >>> from ... import fs
    >>> fs.rand.rng(seed=0, reset=True)
    >>> g = ceg.Graph.new()
    >>> g, ref = g.bind(None, ref=ceg.Ref.Col)
    >>> g, ref = g.bind(
    ...     fs.rand.gaussian.new(ref).sync(
    ...         v=ceg.loop.Fixed(1)
    ...     ),
    ...     ref=ref,
    ...     using=Line.new(title="rand").alias(
    ...         "y"
    ...     ),
    ... )
    >>> g, es = g.steps(ceg.Event(0, ref), n=10)
    >>> g, res = g.flush(es[-1])
    >>> res = render(res)
    >>> {k: type(v) for k, v in res.items()}
    {'rand': <class 'matplotlib.figure.Figure'>}
    """

    def plot(self):
        return core.Line

class Scatter(Continuous_1D):
    """
    >>> from ... import fs
    >>> fs.rand.rng(seed=0, reset=True)
    >>> g = ceg.Graph.new()
    >>> g, ref = g.bind(None, ref=ceg.Ref.Col)
    >>> g, ref = g.bind(
    ...     fs.rand.gaussian.new(ref).sync(
    ...         v=ceg.loop.Fixed(1)
    ...     ),
    ...     ref=ref,
    ...     using=Scatter.new(title="rand").alias(
    ...         "y"
    ...     ),
    ... )
    >>> g, es = g.steps(ceg.Event(0, ref), n=10)
    >>> g, res = g.flush(es[-1])
    >>> res = render(res)
    >>> {k: type(v) for k, v in res.items()}
    {'rand': <class 'matplotlib.figure.Figure'>}
    """

    def plot(self):
        return core.Scatter
        
#  ------------------

# TODO: Lines (many refs)

# exactly like the above, but we assume the unpack y
# is over lists that have been bound to the same key

# x and c are the same

#  ------------------

class Discrete_1D_Kw(NamedTuple):
    scope: ceg.Aliases | None
    grid: Grid_Key
    figure: Optional[str]
    axis: str
    x: str | None = None
    x2: str | None = None
    y: str | None = None
    y2: str | None = None
    c: str | None = None
    slice: int | None = None # or slice
    colors: Optional[core.Color | core.Colors] = None
    format: str | None = None

class Discrete_1D(Discrete_1D_Kw, ceg.Plugin.Aliased):

    @classmethod
    def new(
        cls,
        grid: Grid_Key, # needs all plots already added
        axis: str,
        scope: ceg.Aliases | None = None,
        figure: Optional[str] = None,
        # x / y etc. refer to aliases
        x: Optional[str] = None,
        x2: Optional[str] = None,
        y: Optional[str] = None, 
        y2: Optional[str] = None, 
        # label for series
        c: Optional[str] = None,
        colors: Optional[core.Color | core.Colors] = None,
        format: str | None = None,
    ):
        return cls(
            scope=scope,
            grid=grid,
            axis=axis,
            figure=figure,
            x=x,
            x2=x2,
            y=y,
            y2=y2,
            c=c,
            colors=colors,
            format=format,
        )

    def plot(self) -> Type[core.Discrete_1D]:
        raise ValueError(self)

    def flush(
        self,
        graph: ceg.Graph,
        event: ceg.Event,
        state: ceg.State,
        scope: ceg.Aliases | None,
    ):
        assert isinstance(scope, ceg.Aliases), (
            self,
            scope,
        )
        
        x: list[str] = []
        y: list[float] = []

        c = []

        for (ref, key), kwargs in scope.aliases.items():

            if self.x is not None:
                ref_x = kwargs[self.x]
            elif self.x2 is not None:
                ref_x = kwargs[self.x2]
            else:
                raise ValueError(self)

            if isinstance(ref_x, str):
                v = graph.select(ref, event, t = False)[-1] # type: ignore
                x.append(ref_x)
                y.append(v)
            else:
                x.extend(ref_x)
                y.extend(
                    graph.select(ref, event, t=False)[-1].tolist()
                )
                # type: ignore

        grid: core.Grid = self.grid.get(graph)

        kw = dict(
            x=None if self.x is None else x,
            y=None if self.y is None else y,
            x2=None if self.x2 is None else x,
            y2=None if self.y2 is None else y,
            c=None,
        )

        grid = grid.with_chart(
            # TODO: figure etc.
            getattr(core.fig.axis, self.axis),
            self.plot().new(
                **kw,
                colors=self.colors,
                # format=self.format,
                # TODO: other kwargs?
            ),
        )
        
        return state.set(self.grid, grid)

class Bar(Discrete_1D):

    def plot(self) -> Type[core.Discrete_1D]:
        return core.Bar

#  ------------------

class Discrete_Pairwise_Kw(NamedTuple):
    scope: ceg.Aliases | None
    grid: Grid_Key
    figure: Optional[str]
    axis: str
    x: str | None = None
    y: str | None = None
    c: str | None = None
    slice: int | None = None # or slice
    colors: Optional[core.Color | core.Colors] = None
    format: str | None = None

class Discrete_Pairwise(Discrete_Pairwise_Kw, ceg.Plugin.Aliased):

    @classmethod
    def new(
        cls,
        grid: Grid_Key, # needs all plots already added
        axis: str,
        scope: ceg.Aliases | None = None,
        figure: Optional[str] = None,
        # x / y etc. refer to aliases
        x: Optional[str] = None,
        y: Optional[str] = None, # label for series
        c: Optional[str] = None,
        colors: Optional[core.Color | core.Colors] = None,
        format: str | None = None,
    ):
        return cls(
            scope=scope,
            grid=grid,
            axis=axis,
            figure=figure,
            x=x,
            y=y,
            c=c,
            colors=colors,
            format=format,
        )

    def plot(self) -> Type[core.Discrete_Pairwise]:
        raise ValueError(self)

    def flush(
        self,
        graph: ceg.Graph,
        event: ceg.Event,
        state: ceg.State,
        scope: ceg.Aliases | None,
    ):
        assert isinstance(scope, ceg.Aliases), (
            self,
            scope,
        )
        cells = []
        x: list[str] = []
        y: list[str] = []
        c = []

        for (ref, key), kwargs in scope.aliases.items():
            cells.append(key)

            assert self.x is not None, self
            x.append(kwargs[self.x])

            assert self.y is not None, self
            y.append(kwargs[self.y])

            v = graph.select(ref, event, t = False)[-1] # type: ignore
            c.append(v)

        grid: core.Grid = self.grid.get(graph)

        grid = grid.with_chart(
            # TODO: figure etc.
            getattr(core.fig.axis, self.axis),
            self.plot().new(
                x,
                y,
                c,
                colors=self.colors,
                format=self.format,
                # TODO: other kwargs?
            ),
        )
        
        return state.set(self.grid, grid)

class Heatmap(Discrete_Pairwise):

    def plot(self) -> Type[core.Discrete_Pairwise]:
        return core.Rectangle

#  ------------------
