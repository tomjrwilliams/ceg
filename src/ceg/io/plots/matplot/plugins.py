from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import NamedTuple, Any, Optional, Type
from collections import defaultdict

from frozendict import frozendict

import numpy as np

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

def unpack_aliases_1d_y(
    mark: Continuous_1D | Continuous_2D,
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
    t: str | None = None,
    c: str | None = None,
    label: str | None = None,
    shift: float | None = None,
    prev: ceg.Event | None = None,
    transform: str | None = None,
):
    x = x2 = c = None
    # where = None if prev is None else dict(t=lambda t: t >= prev.t)
    where = None
    tt, y = graph.select(ref, at=event, t=True, where=where)
    y_nan = np.isnan(y)
    if shift is not None:
        i_shift = int(abs(shift))
        y_not_nan = y[np.logical_not(y_nan)]
        if shift > 0:
            y[np.logical_not(y_nan)] = np.concatenate((np.ones(i_shift) * np.NAN, y_not_nan[:-i_shift]))
        else:
            y[np.logical_not(y_nan)] = np.concatenate((y_not_nan[i_shift:], np.ones(i_shift) * np.NAN))
    if prev is not None:
        y = y[tt >= prev.t]
        tt = tt[tt >= prev.t]
    y_nan = np.isnan(y)
    if transform == "cum":
        y = np.nan_to_num(y, nan=0)
        y = np.cumsum(y)
        y[y_nan] = np.NAN
    elif transform == "commpound":
        y = np.nan_to_num(y, nan=0)
        y = np.cumprod(1 + y)
        y[y_nan] = np.NAN
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

def unpack_aliases_1d_x(
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
    shift: float | None = None,
    prev: ceg.Event | None = None,
):
    where = None if prev is None else dict(t=lambda t: t>=prev.t)
    x = graph.select(ref, at=event, where=where)
    if isinstance(slice, (int, t_slice)):
        x = x[slice]
    elif isinstance(ref, ceg.Ref.Col):
        raise ValueError(ref)
    elif isinstance(ref, ceg.Ref.Col1D):
        x = x[-1]
    return x

def unpack_aliases_1d_c(
    ref: ceg.Ref.Col | ceg.Ref.Col1D,
    graph: ceg.Graph,
    event: ceg.Event,
    slice: int | slice | None=None,
    shift: float | None = None,
    prev: ceg.Event | None = None,
):
    t, x = graph.select(ref, at=event, t=True)
    if prev is not None:
        x = x[t >= prev.t]
    if isinstance(slice, (int, t_slice)):
        x = x[slice]
    elif isinstance(ref, ceg.Ref.Col):
        raise ValueError(ref)
    elif isinstance(ref, ceg.Ref.Col1D):
        x = x[-1]
    return x

def unpack_aliases_1d_xyc(
    mark: Continuous_1D,
    aliases: frozendict[
        str, tuple[ceg.Ref.Any, frozendict]
    ],
    graph: ceg.Graph,
    event: ceg.Event,
    prev: ceg.Event | None = None,
):
    x = None
    y = None
    x2 = None
    y2 = None
    c = None

    if mark.y is not None and mark.y in aliases:
        ref, kwargs = aliases[mark.y]
        y, x, x2, c = unpack_aliases_1d_y(
            mark,
            ref, # type: ignore
            graph, event, prev=prev,**kwargs
        )

    if mark.y2 is not None and mark.y2 in aliases:
        ref, kwargs = aliases[mark.y2]
        y2, x, x2, c = unpack_aliases_1d_y(
            mark,
            ref, # type: ignore
            graph, event, prev=prev, **kwargs
        )

    assert y is not None or y2 is not None, aliases

    return x, y, x2, y2, c

def unpack_aliases_continuous_1d(
    mark: Continuous_1D,
    aliases: frozendict[
        str, tuple[ceg.Ref.Any, frozendict]
    ],
    graph: ceg.Graph,
    event: ceg.Event,
    prev: ceg.Event | None = None,
):
    x, y, x2, y2, c = unpack_aliases_1d_xyc(
        mark, aliases, graph, event, prev=prev
    )

    if x is None and mark.x is not None and mark.x in aliases:
        ref, kwargs = aliases[mark.x]
        x = unpack_aliases_1d_x(
            ref,  # type: ignore
            graph, event, prev=prev, **kwargs
        )

    if x2 is None and mark.x2 is not None and mark.x2 in aliases:
        ref, kwargs = aliases[mark.x2]
        x2 = unpack_aliases_1d_x(
            ref,  # type: ignore
            graph, event, prev=prev, **kwargs)

    if c is None and mark.c is not None and mark.c in aliases:
        ref, kwargs = aliases[mark.c]
        c = unpack_aliases_1d_c(
            ref, # type: ignore
            graph,
            event, prev=prev,
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

def unpack_aliases_2d_yc(
    mark: Continuous_2D,
    aliases: frozendict[
        str, list[tuple[ceg.Ref.Any, frozendict]]
    ],
    graph: ceg.Graph,
    event: ceg.Event,
    prev: ceg.Event | None = None,
):
    y = []
    y2 = []
    c = []

    # TODO: if any have t=True, assert all do?
    # for now always return none for x, x2

    if mark.y is not None and mark.y in aliases:
        for ref, kwargs in aliases[mark.y]:
            y_, _, _, c_ = unpack_aliases_1d_y(
                mark,
                ref, # type: ignore
                graph, event, prev=prev, **kwargs
            )
            y.append(y_)
            if c_ is not None:
                c.append(c_)

    if mark.y2 is not None and mark.y2 in aliases:
        for ref, kwargs in aliases[mark.y2]:
            y2_, _, _, c_ = unpack_aliases_1d_y(
                mark,
                ref, # type: ignore
                graph, event, prev=prev, **kwargs
            )
            y2.append(y2_)
            if c_ is not None:
                c.append(c_)

    if not len(y):
        y = None
    if not len(y2):
        y2 = None
    if not len(c):
        c = None

    return y, y2, c

REF_KWARGS = tuple[ceg.Ref.Any, frozendict]

def unpack_aliases_continuous_2d(
    mark: Continuous_2D,
    aliases: frozendict[
        str, REF_KWARGS | list[REF_KWARGS]
    ],
    graph: ceg.Graph,
    event: ceg.Event,
    prev: ceg.Event | None = None,
):
    y, y2, c = unpack_aliases_2d_yc(
        mark,
        aliases, # type: ignore
        graph, event, prev=prev, 
    )

    x = None
    x2 = None

    if x is None and mark.x is not None and mark.x in aliases:
        ref, kwargs = aliases[mark.x]
        x = unpack_aliases_1d_x(
            ref,  # type: ignore
            graph, event, prev=prev,
            **kwargs # type: ignore
        )

    if x2 is None and mark.x2 is not None and mark.x2 in aliases:
        ref, kwargs = aliases[mark.x2]
        x2 = unpack_aliases_1d_x(
            ref,  # type: ignore
            graph, event, prev=prev,
            **kwargs # type: ignore
        )

    if c is None and mark.c is not None and mark.c in aliases:
        ref, kwargs = aliases[mark.c]
        c = unpack_aliases_1d_c(
            ref, # type: ignore
            graph,
            event, prev=prev,
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

#  ------------------


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
        prev: ceg.Event | None = None,
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

        kwargs = unpack_aliases_continuous_1d(
            self,
            aliases, # type: ignore
            graph, 
            event,
            prev=prev,
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


class Continuous_2D_Kw(NamedTuple):
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
    window: Optional[float] = None
    consts: frozendict[str, float] | None = None

class Continuous_2D(Continuous_2D_Kw, ceg.Plugin.Aliased):

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
        consts: frozendict[str, float] | None = None,
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
            consts=consts,
        )

    def plot(self) -> Type[core.Continuous_2D]:
        raise ValueError()

    def flush(
        self,
        graph: ceg.Graph,
        event: ceg.Event,
        state: ceg.State,
        scope: ceg.Aliases | None,
        prev: ceg.Event | None = None,
    ):
        assert isinstance(scope, ceg.Aliases), (
            self,
            scope,
        )

        keys = list(set(key for (_, key) in scope.aliases.keys()))
        keys_2d = [
            k for k in keys if k in {self.y, self.y2, self.c}
        ]

        aliases: dict[str, REF_KWARGS | list[REF_KWARGS]] = {
            k: [] for k in keys_2d
        }

        labels_y = []
        labels_y2 = []

        for (ref, key), kwargs in scope.aliases.items():
            label = kwargs.get("label")
            if key in keys_2d:
                ref_kwargs = aliases[key]
                assert isinstance(ref_kwargs, list), aliases
                ref_kwargs.append((ref, kwargs))
                if key == self.y:
                    labels_y.append(label)
                if key == self.y2:
                    labels_y2.append(label)
            else:
                assert key not in aliases, key
                aliases[key] = (ref, kwargs)

        kwargs = unpack_aliases_continuous_2d(
            self,
            frozendict(aliases), # type: ignore
            graph,
            event,
            prev=prev,
        )

        grid: core.Grid = self.grid.get(graph)
    
        kw_y = [dict(label=l) for l in labels_y]
        kw_y2 = [dict(label=l) for l in labels_y2]
        hidden=[
            kw["label"] for kw in kw_y
        ] # TODO: is being popped in the call below?

        kws = {"y": kw_y, "y2": kw_y2}

        for ykey in ["y", "y2"]:
            if self.consts is not None and kwargs[ykey] is not None:
                y_ex = kwargs[ykey][0]
                for k, v in self.consts.items():
                    kwargs[ykey].append([
                        v for _ in y_ex
                    ])
                    kws[ykey].append(
                        dict(label=k, color="black", linestyle=(0, (1, 5)))
                    )
        
        y = kwargs.pop("y")
        y2 = kwargs.pop("y2")

        # we split y then y2 in core
        # this would all be easier if we use the kwargs not the key?
        # more like the discrete logic
        try:
            if y is not None:
                grid = grid.with_chart(
                    # TODO: figure etc.
                    getattr(core.fig.axis, self.axis),
                    self.plot().new(
                        **kwargs,
                        y=y,
                        colors=self.colors,
                        kwargs = kws["y"]
                    ),
                )
            if y2 is not None:
                ax = getattr(core.fig.axis, self.axis).twin
                grid = grid.with_chart(
                    # TODO: figure etc.
                    ax,
                    self.plot().new(
                        **kwargs,
                        y2=y2,
                        colors=self.colors,
                        kwargs = kws["y2"],
                        hidden=hidden,
                    ),
                )
        except:
            raise ValueError(self)
        
        return state.set(self.grid, grid)


#  ------------------

class Lines(Continuous_2D):
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
    ...     using=Lines.new(title="rand").alias(
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
        return core.Lines

class Scatters(Continuous_2D):
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
    ...     using=Scatters.new(title="rand").alias(
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
        return core.Scatters
        

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
        prev: ceg.Event | None = None,
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
                assert isinstance(ref, ceg.Ref.Col1D), ref
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
        prev: ceg.Event | None = None,
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

            v = graph.select(ref, event, t = False, i=-1, null=False) # type: ignore
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
