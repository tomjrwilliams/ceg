from __future__ import annotations
from typing import Optional, NamedTuple, Union, Callable, cast

from frozendict import frozendict

from functools import lru_cache

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
import matplotlib.figure
import matplotlib.axis
import matplotlib.axes
import matplotlib.dates

from matplotlib import pyplot as plt

# -----------------

EMPTY = "EMPTY"

FigureOrSubFigure = Union[
    matplotlib.figure.Figure,
    matplotlib.figure.SubFigure,
]

# -----------------

class AxisRef(NamedTuple):
    fig: FigureRef
    axis: Optional[str]
    twinned: bool

    def with_grid(self, grid: Grid) -> AxisRef:
        return self._replace(
            fig=self.fig.with_grid(grid)
        )

    def __getattr__(
        self, name: str
    ) -> AxisRef:
        return self._replace(axis=name)
    
    @property
    def twin(self) -> AxisRef:
        return self._replace(twinned=True)

    @property
    def obj(self) -> matplotlib.axes.Axes | TwinnedAxes:
        fig = self.fig
        assert fig.grid is not None, fig
        grid = fig.grid
        if grid.figures is not None:
            assert fig.figure is not None, (
                list(grid.figures.keys()),
                fig.figure
            )
            if self.twinned:
                axes = grid.twins[fig.figure]
            else:
                axes = grid.axes[fig.figure]
            if self.axis is None:
                assert isinstance(
                    axes, (matplotlib.axes.Axes, TwinnedAxes)
                )
                return axes
            assert isinstance(axes, frozendict), axes
            return axes[self.axis]
        assert fig.figure is None, (
            grid.figures,
            fig.figure
        )
        assert self.axis is not None, self
        if self.twinned and self.axis:
            res = grid.twins[self.axis]
            assert isinstance(res, matplotlib.axes.Axes)
        else:
            res = grid.axes[self.axis]
        assert isinstance(
            res, (matplotlib.axes.Axes, TwinnedAxes)
        )
        return res

class FigureRef(NamedTuple):
    grid: Optional[Grid]
    figure: Optional[str]
    
    def with_grid(self, grid: Grid) -> FigureRef:
        return self._replace(grid=grid)

    @property
    def obj(self) -> FigureOrSubFigure:
        assert self.grid is not None, self
        grid = self.grid
        if self.figure is None:
            assert grid.figures is None, (
                grid.figures, self.figure
            )
            return grid.figure
        assert grid.figures is not None, grid.figures
        return grid.figures[self.figure]
    
    @property
    def axes(self) -> frozendict[
        str, matplotlib.axes.Axes
    ]:
        assert self.grid is not None, self
        grid = self.grid
        assert grid.axes is not None, grid
        if self.figure is None:
            assert grid.figures is None, (
                grid.figures, self.figure
            )
            return cast(
                frozendict[str, matplotlib.axes.Axes],
                grid.axes
            )
        res = grid.axes[self.figure]
        assert isinstance(res, frozendict), res
        # are we sure? don't sometimes want singular? no, should be via figure
        return res

    @property
    def axis(self) -> AxisRef:
        return AxisRef(self, None, False)

    def __getattr__(
        self, name: str
    ) -> FigureRef:
        return self._replace(figure=name)

fig = FigureRef(None, None)

# -----------------

def twin_axes(
    axes: frozendict[str, matplotlib.axes.Axes],
    twin_x: Union[bool, list[str]] = False,
    twin_y: Union[bool, list[str]] = False,
):
    if twin_x == True and twin_y == True:
        return frozendict({
            label: TwinnedAxes(
                ax.twinx().twiny(),
                True,
                True,
            )
            for label, ax in axes.items()
        })
    if twin_x == False:
        twin_x = []
    elif twin_x == True:
        twin_x = list(axes.keys())
    if twin_y == False:
        twin_y = []
    elif twin_y == True:
        twin_y = list(axes.keys())
    return frozendict({
        label: (
            TwinnedAxes(
                ax.twinx().twiny(),
                True,
                True,
            )
            if label in twin_y and label in twin_x
            else TwinnedAxes(
                ax.twiny(),
                False,
                True,
            )
            if label in twin_y
            else TwinnedAxes(
                ax.twinx(),
                True,
                False,
            )
            if label in twin_x
            else TwinnedAxes(ax, False, False)
        )
        for label, ax in axes.items()
    })

class TwinnedAxes(NamedTuple):
    axes: matplotlib.axes.Axes
    x: bool
    y: bool

# -----------------

from .... import core as ceg

class Grid_Key_Kw(NamedTuple):
    name: str

class Grid_Key(Grid_Key_Kw, ceg.Key):
    
    def get(self, graph: ceg.Graph):
        res: ceg.Value | None = graph.state.get(self)
        assert res is not None, self
        assert isinstance(res, Grid), self
        return res

class Grid_Kw(NamedTuple):

    figure: matplotlib.figure.Figure
    figures: Optional[
        frozendict[str, matplotlib.figure.SubFigure]
    ]
    axes: frozendict[
        str, Union[
            matplotlib.axes.Axes,
            frozendict[str, matplotlib.axes.Axes]
        ]
    ]
    # TODO: specify which is twinned so we can check
    twins: frozendict[
        str, Union[
            TwinnedAxes,
            frozendict[str, TwinnedAxes]
        ]
    ]
    charts: tuple[Mark, ...]
    dfs: frozendict[str, pd.DataFrame]

class Grid(Grid_Kw, ceg.Value):

    figure: matplotlib.figure.Figure
    figures: Optional[
        frozendict[str, matplotlib.figure.SubFigure]
    ]
    axes: frozendict[
        str, Union[
            matplotlib.axes.Axes,
            frozendict[str, matplotlib.axes.Axes]
        ]
    ]
    # TODO: specify which is twinned so we can check
    twins: frozendict[
        str, Union[
            TwinnedAxes,
            frozendict[str, TwinnedAxes]
        ]
    ]
    charts: tuple[Mark, ...]
    dfs: frozendict[str, pd.DataFrame | pl.DataFrame]

    @property
    def fig(self) -> FigureRef:
        return FigureRef(self, None)

    @classmethod
    def new(
        cls,
        layout: str = "constrained",
        figsize: tuple[float, float] | None = None,
        **kwargs
    ) -> Grid:
        fig = plt.figure(
            layout=layout,
            figsize=figsize,
            **kwargs,
        )
        return cls(
            figure=fig,
            figures=None,
            axes=frozendict(),  #type: ignore
            twins=frozendict(),  #type: ignore
            charts=(),
            dfs=frozendict()  #type: ignore
        )
    
    def with_sub_figures(
        self,
        labels: list[list[str]],
        width_ratios: Optional[tuple[float, ...]] = None,
        height_ratios: Optional[tuple[float, ...]] = None,
    ):
        all_labels = sum(labels, [])
        assert len(all_labels) == len(set(all_labels)), all_labels
        assert self.figures is None, self
        n_rows = len(labels)
        n_cols = [len(r) for r in labels]
        assert all((
            n == n_cols[0] for n in n_cols
        )), (n_cols)
        f_figures = lambda: self.figure.subfigures(
            nrows=n_rows,
            ncols=n_cols[0],
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        if n_rows == 1 and n_cols == 1:
            sub_figures = [[f_figures()]]
        elif n_rows == 1:
            sub_figures = [f_figures()]
        elif n_cols == 1:
            sub_figures = [
                [sub] for sub in f_figures()
            ]
        else:
            sub_figures = f_figures()
        self = self._replace(figures={
            label: sub_fig
            for label_row, fig_row
            in zip(labels, sub_figures)
            for label, sub_fig
            in zip(label_row, fig_row)
        })
        assert self.figures is not None, self
        for label, sub_fig in self.figures.items():
            sub_fig.suptitle(label)
        return self

    def with_axes(
        self,
        labels: list[list[str]],
        sharex: bool=False,
        sharey: bool=False,
        figure: Optional[str] = None,
        subplot_kw: dict = {},
        gridspec_kw: dict = {},
        twin_x: Union[bool, list[str]] = False,
        twin_y: Union[bool, list[str]] = False,
        empty_marker: str = EMPTY,
    ) -> Grid:
        if self.figures is None:
            assert figure is None, self
            fig = self.figure
        else:
            assert figure is not None, list(
                self.figures.keys()
            )
            assert figure not in self.axes, (
                list(self.axes.keys()), figure
            )
            fig = self.figures[figure]
        axes = frozendict(fig.subplot_mosaic(
            labels, # type: ignore
            sharex=sharex,
            sharey=sharey,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            empty_sentinel=empty_marker,
        ))
        twins = twin_axes(
            axes,
            twin_x,
            twin_y,
        )
        for label, axis in axes.items():
            axis: matplotlib.axes.Axes
            axis.set_title(label, loc="left")
        if figure is None:
            return self._replace(axes=axes, twins=twins)
        return self._replace(
            axes=self.axes.set(
                figure, axes
            ),
            twins=self.twins.set(
                figure, twins
            )
        )
    
    def with_chart(
        self,
        axis: AxisRef,
        chart: Mark,
    ) -> Grid:
        axis_ref = axis.with_grid(self)
        chart = chart._replace(
            figure=axis_ref.fig.figure,
            axis=axis_ref.axis,
        )
        chart.apply(axis_ref, self)
        return self._replace(
            charts = self.charts + (chart,)
        )
    
    def with_data(
        self, **dfs: pd.DataFrame,
    ) -> Grid:
        curr = self.dfs
        for k, df in dfs.items():
            curr = curr.set(k, df)
        return self._replace(dfs=curr)

    def unpack_col(
        self: Grid,
        col: ArrayOrCol,
        data: Optional[str] = None,
        axis: Optional[np.ndarray | list] = None,
    ) -> np.ndarray | list:
        if isinstance(col, (int, float)):
            assert axis is not None, (col, axis)
            return [col for _ in axis]
        if isinstance(col, str):
            assert data is not None, (col, data)
            df = self.dfs[data]
            if isinstance(df, pd.DataFrame):
                return df[col].to_numpy()
            assert isinstance(df, pl.DataFrame), df
            return df.get_column(col).to_numpy()
            
        assert isinstance(col, (list, np.ndarray)), col
        return col
    
    def unpack_cols(
        self: Grid,
        cols: ArrayOrCols,
        data: Optional[str] = None,
        axis: Optional[np.ndarray | list] = None,
    ) -> np.ndarray | list:
        if isinstance(cols, np.ndarray):
            return [
                cols[:, i] # type: ignore
                for i in range(cols.shape[1])
            ]
        assert isinstance(cols, list), cols
        return [
            self.unpack_col(c, data=data,axis =axis)
            for c in cols
        ]
    
    def show(self):
        try:
            plt.figure(self.figure)
            plt.show()
        finally:
            plt.clf()

    def write(self, f, figsize = None):
        try:
            plt.figure(
                self.figure,
                figsize=figsize
            )
            # or fp?
            plt.savefig(f, bbox_inches='tight')
            plt.close()
        finally:
            plt.clf()


# -----------------

ArrayOrCol = Union[int, float, str, list, np.ndarray]
ArrayOrCols = Union[
    list[ArrayOrCol],
    list[np.ndarray],
    #
]

def cols_shape(
    y: Optional[ArrayOrCols]=None,
    y2: Optional[ArrayOrCols]=None,
) -> tuple[int, ...] | None:
    v = (
        y if y is not None
        else y2 if y2 is not None
        else None
    )
    if v is None:
        return None
    # shape[-1] is the number of lines (cols)
    if isinstance(v, list):
        return (len(v),)
    assert isinstance(v, np.ndarray), type(v)
    return v.shape

# -----------------

# https://matplotlib.org/stable/users/explain/colors/colormaps.html

# ListedColormap has .colors and is callable
# ListedSegemnted.. is just callable with unit range values

# create Listed directly: ListedColormap(colors)
# assumed to evenly span, discrete nearest neighbour
# for interpolation, use Segmented

# create ListedSegmented... by passing a list of unit interval values and a list of colors
# diff is it interpolates rgba between break

import matplotlib.colors

@lru_cache(maxsize=128)
def color_map(
    name: str,
    n: Optional[int]=None,
    reversed: bool = False,
):
    cmap = matplotlib.colormaps[name]
    if n is not None:
        cmap = cmap.resampled(n)
    if reversed:
        cmap = cmap.reversed()
    return cmap

def cmap_unique_colors(
    cmap: Union[
        matplotlib.colors.ListedColormap,
        matplotlib.colors.LinearSegmentedColormap,
        matplotlib.colors.Colormap
    ],
    n: int
):
    cmap = cmap.resampled(n)
    colors: np.ndarray = cmap.__dict__["colors"]
    rgba = map(tuple(colors.tolist())) # type: ignore
    return len(set(rgba))

@lru_cache(maxsize=128)
def cmap_type_arity(
    key: str,
):
    arity = None
    try:
        cmap = matplotlib.colormaps[key]
    except:
        raise ValueError(f"Invalid cmap: {key}")
    type=(
        "discrete"
        if isinstance(
            cmap, 
            matplotlib.colors.ListedColormap
        )
        else "segmented"
        if isinstance(
            cmap,
            matplotlib.colors.LinearSegmentedColormap
        )
        else None,
    )
    if type == "discrete":
        n_50 = cmap_unique_colors(cmap, 50)
        n_200 = cmap_unique_colors(cmap, 200)
        if n_50 != n_200:
            type == "continuous"
        else:
            arity = n_50
    return type, arity

class Colors(NamedTuple):
    type: Optional[str]
    arity: Optional[int]
    cmap: Optional[str]
    start: Optional[float]
    end: Optional[float]
    bins: Optional[int]
    reversed: bool
    # eg. continous map
    # discretised up front to n colors

    @classmethod
    def new(
        cls,
        type: Optional[str]=None,
        arity: Optional[int] = None,
        cmap: Optional[str]=None,
        start: Optional[float]=None,
        end: Optional[float]=None,
        bins: Optional[int] = None,
        reversed: bool = False,
    ):
        return cls(
            type=type,
            arity=arity,
            cmap=cmap,
            start=start,
            end=end,
            bins=bins,
            reversed=reversed,
        )
    
    def reverse(self):
        assert not self.reversed, self
        return self._replace(reversed=True)
    
    def with_bins(
        self, bins: int
    ) -> Colors:
        assert self.bins is None, self
        return self._replace(bins=bins)

    def with_range(self, start: float, end: float):
        return self._replace(start=start, end=end)

    def range(self):
        l = 0 if self.start is None else self.start
        r = 1 if self.end is None else self.end
        return l, r

    def n_colors(self):
        bins = self.bins
        if bins is None and self.type == "discrete":
            bins = self.arity
        return bins

    def boundaries(self):
        n = self.n_colors()
        assert n is not None, self
        l, r = self.range()
        width = (r - l) / n
        return np.array([
            l + width + (i * width)
            for i in range(n)
        ])

    def sample(
        self, cval: Union[int, float]
    ) -> Color:
        bins = self.n_colors()
        if isinstance(cval, int):
            assert bins is not None, (cval, self)
            if cval > bins:
                raise ValueError(cval, self)
            cval = cval / bins
        cmap = color_map(
            self.cmap,
            n=bins,
            reversed=self.reversed,
        )
        l, r = self.range()
        cval_unit = (cval - l) / (r-l)
        rgba = cmap(cval_unit)
        return Color.new(rgba=rgba).with_hex()

    def __getattr__(self, key: str) -> Colors:
        assert self.cmap is None, self
        type, arity = cmap_type_arity(key)
        return self._replace(
            type=type,
            arity=arity,
            cmap=key
        )

class Color(NamedTuple):
    name: Optional[str]
    rgba: Optional[tuple[float, float, float, float]]
    hex: Optional[str]

    def color(self) -> str:
        if self.name:
            return self.name
        if self.hex:
            return self.hex
        assert self.rgba is not None, self
        s = self.with_hex().hex
        assert s is not None, self
        return s

    def with_hex(self) -> Color:
        assert self.rgba is not None, self
        return self._replace(
            hex=matplotlib.colors.rgb2hex(
                self.rgba, keep_alpha=True
            )
        )

    @classmethod
    def new(
        cls,
        name: Optional[str]=None,
        rgba: Optional[tuple[float, float, float, float]]=None,
        hex: Optional[str]=None,
    ):
        return cls(
            name=name,
            rgba=rgba,
            hex=hex,
        )

colors = Colors.new()

def color(name: str) -> Color:
    # TODO: with_hex / rgba
    return Color.new(name=name)

def hex(s: str) -> Color:
    # TODO: with rgba
    return Color.new(hex=s)

def rgba(
    arg: Union[
        float, tuple[float, float, float, float]
    ],
    *args: float
) -> Color:
    if isinstance(arg, tuple):
        assert len(arg) == 4, arg
        assert len(args) == 0, args
        rgba = arg
    else:
        assert isinstance(arg, float), arg
        args = (arg,) + args
        assert len(args) == 4, args
        rgba = args
    return Color.new(rgba=rgba).with_hex()

# -----------------

class Mark_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    
class Mark(Mark_Kw):
    
    def apply(
        self,
        axis: AxisRef,
        grid: Grid,
    ):
        raise ValueError(self)

# -----------------

class Discrete_Pairwise_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    x: list[str]
    y: list[str]
    c: ArrayOrCol
    colors: Optional[Union[Color, Colors]]
    format: str | None
    data: Optional[str]
    kwargs: dict

class Discrete_Pairwise(Discrete_Pairwise_Kw, Mark):

    def plot(
        self, axes: matplotlib.axes.Axes
    ) -> Callable:
        raise ValueError(self)

    @classmethod
    def new(
        cls, 
        x: list[str],
        y: list[str],
        c: ArrayOrCol,
        data: Optional[str] = None,
        colors: Optional[Union[Color, Colors]]=None,
        format: str | None = None,
        **kwargs
    ):
        return cls(
            figure=None,
            axis=None,
            x=x,
            y=y,
            c=c,
            colors=colors,
            format=format,
            data=data,
            kwargs=kwargs,
        )

    def apply(
        self,
        axis: AxisRef,
        grid: Grid,
    ):
        assert isinstance(axis.obj, matplotlib.axes.Axes)
        ax = axis.obj
        x = self.x
        y = self.y
        assert isinstance(self.colors, Colors), self
        c = grid.unpack_col(
            self.c, data=self.data, axis=x,
        )
        if self.format == "pct":
            c = [cc * 100 for cc in c]
        cs = [
            None if c == np.NaN
            else self.colors.sample(cc).with_hex().hex
            for cc in c
        ]
        # TODO: widths, lengths - possibly via x2, y2?

        xs = {}
        ys = {}
        xy_cs = {}
        xy_vs = {}

        plot = self.plot(ax)

        # TODO: assert same length
        
        for xx, yy, cc, v in zip(x, y, cs, c):
            xs[xx] = None
            ys[yy] = None
            xy_cs[(xx, yy)] = cc
            xy_vs[(xx, yy)] = v

        xrange = np.array(range(len(xs)))
        yrange = np.array(range(len(ys)))
        
        ax.scatter(xrange + 1, yrange + 1, color="white")

        for ix, xx in enumerate(xs.keys()):
            for iy, yy in enumerate(ys.keys()):

                cc = xy_cs.get((xx, yy), None)
                v = xy_vs.get((xx, yy), None)

                if cc is None:
                    continue

                if self.format == "pct" and isinstance(
                    v, (int, float)
                ):
                    v = f"{round(v, 3)}%"

                plot(
                    ix,
                    iy,
                    color=cc,
                    xlabel=xx,
                    ylabel=yy,
                    text=v,
                )
        
        ax.set_xticks(xrange + .5, list(xs.keys()))
        ax.set_yticks(yrange + .5, list(ys.keys()))

    
import matplotlib.patches

class Rectangle(Discrete_Pairwise):

    def plot(self, axes: matplotlib.axes.Axes):
        def f(ix, iy, color, xlabel, ylabel, text):
            width = 1
            height = 1
            patch = matplotlib.patches.Rectangle(
                (ix, iy),
                width,
                height,
                color=color,
                fill=True,
            )

            if isinstance(color, tuple):
                rgb = color
            else:
                rgb = matplotlib.colors.hex2color(color)

            grey = sum(rgb) / len(rgb)

            text_color = "black" if grey > 0.5 else "white"

            # w_per_char = (width / 2) / len(text)
            # offset = int(len(text) / 2) * w_per_char

            axes.annotate(
                text,
                (
                    ix + 0.1 * width,
                    iy + 0.1 * height
                ), 
                color=text_color
            )
            axes.add_patch(patch)
        return f

# -----------------

class Discrete_1D_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    x: Optional[list[str]]
    x2: Optional[list[str]]
    y: Optional[ArrayOrCol]
    y2: Optional[ArrayOrCol]
    # c tbc?
    c: Optional[ArrayOrCol]
    colors: Optional[Union[Color, Colors]]
    data: Optional[str]
    kwargs: dict

class Discrete_1D(Discrete_1D_Kw, Mark):

    def plot(
        self, axes: matplotlib.axes.Axes
    ) -> Callable:
        raise ValueError(self)

    @classmethod
    def new(
        cls, 
        x: Optional[list[str]]=None,
        y: Optional[ArrayOrCol]=None,
        x2: Optional[list[str]]=None,
        y2: Optional[ArrayOrCol]=None,
        c: Optional[ArrayOrCol]=None,
        data: Optional[str] = None,
        colors: Optional[Union[Color, Colors]]=None,
        **kwargs
    ):
        assert x is None or x2 is None, (x, x2)
        assert y is None or y2 is None, (y, y2)
        return cls(
            figure=None,
            axis=None,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            c=c,
            colors=colors,
            data=data,
            kwargs=kwargs,
        )

    def apply(
        self,
        axis: AxisRef,
        grid: Grid,
    ):
        if self.x is None and self.y is None:
            assert self.x2 is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and y_twin, axis
            x = self.x2
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        elif self.x is None:
            assert self.x2 is not None, self
            assert self.y is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and not y_twin, axis
            x = self.x2
            y = grid.unpack_col(
                self.y, data=self.data
            )
        elif self.y is None:
            assert self.x is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert y_twin and not x_twin, axis
            x = self.x
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        else:
            assert isinstance(axis.obj, matplotlib.axes.Axes)
            ax = axis.obj
            x = self.x
            y = grid.unpack_col(
                self.y, data=self.data
            )

        plot = self.plot(ax)
        plot(
            x,
            y,
            # color=c, # TODO: colors mapping?
            **self.kwargs,
        )

class Bar(Discrete_1D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.bar
        
# -----------------

class Continuous_1D_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    x: Optional[ArrayOrCol]
    y: Optional[ArrayOrCol]
    x2: Optional[ArrayOrCol]
    y2: Optional[ArrayOrCol]
    c: Optional[ArrayOrCol]
    colors: Optional[Union[Color, Colors]]
    data: Optional[str]
    kwargs: dict

class Continuous_1D(Continuous_1D_Kw, Mark):

    def plot(
        self, axes: matplotlib.axes.Axes
    ) -> Callable:
        raise ValueError(self)

    @classmethod
    def new(
        cls, 
        x: Optional[ArrayOrCol]=None,
        y: Optional[ArrayOrCol]=None,
        x2: Optional[ArrayOrCol]=None,
        y2: Optional[ArrayOrCol]=None,
        c: Optional[ArrayOrCol]=None,
        data: Optional[str] = None,
        colors: Optional[Union[Color, Colors]]=None,
        **kwargs
    ):
        assert x is None or x2 is None, (x, x2)
        assert y is None or y2 is None, (y, y2)
        return cls(
            figure=None,
            axis=None,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            c=c,
            colors=colors,
            data=data,
            kwargs=kwargs,
        )

    def apply(
        self,
        axis: AxisRef,
        grid: Grid,
    ):
        if self.x is None and self.y is None:
            assert self.x2 is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        elif self.x is None:
            assert self.x2 is not None, self
            assert self.y is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and not y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_col(
                self.y, data=self.data
            )
        elif self.y is None:
            assert self.x is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert y_twin and not x_twin, axis
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        else:
            assert isinstance(axis.obj, matplotlib.axes.Axes)
            ax = axis.obj
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_col(
                self.y, data=self.data
            )
        color = None
        if isinstance(self.colors, Color):
            color = self.colors.color()
        plt.setp(
            ax.get_xticklabels(),
            rotation=45, 
            horizontalalignment='center'
        )
        for xx in x:
            if xx != np.NAN:
                if isinstance(xx, dt.date):
                    date_format_x(ax, x, xx)
                break
        if self.c is None:
            assert not isinstance(
                self.colors, Colors
            ), self.colors
            return self.plot(ax)(
                x,
                y,
                color=color,
                **self.kwargs
            )
        assert isinstance(self.colors, Colors), self
        boundaries = self.colors.boundaries()
        # TODO: fill boundaries with value range
        # if not given
        c = grid.unpack_col(
            self.c, data=self.data, axis=x,
        )
        boundaried = boundaries[
            np.digitize(
                c, boundaries
            ) - 1 
            # apparently start=1 indexed?
        ]
        inds = np.linspace(0, 1, len(c))
        segs = np.split(
            inds,
            np.where(
                boundaried[1:] 
                != boundaried[:-1]
            )
            # + 1 ?
        )
        # TODO: warn if more than a certain ratio of segments to length
        plot = self.plot(ax)
        for seg in segs:
            c = c[seg][0]
            plot(
                x[seg],
                y[seg],
                color=c,
                **self.kwargs,
            )
        return

def date_format_x(ax, x, d):
    fmt = (
        "%Y-%m" if len(x) >= 90
        else "%m-%d"
    )
    if len(x) < 90:
        ax.set_title(
            str(ax.title) + " " + str(d.year)
        )
    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter(fmt)
    )

# -----------------

class Scatter(Continuous_1D):
    
    def plot(self, axes: matplotlib.axes.Axes):
        return axes.scatter

class Line(Continuous_1D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.plot
    
# -----------------

class Continuous_2D_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    x: Optional[ArrayOrCol]
    y: Optional[ArrayOrCols]
    x2: Optional[ArrayOrCol]
    y2: Optional[ArrayOrCols]
    c: Optional[ArrayOrCols]
    data: Optional[str]
    colors: Optional[Union[Color, Colors]]
    kwargs: Optional[list[dict]]
    shared: dict


class Continuous_2D(Continuous_2D_Kw, Mark):

    def plot(
        self, axes: matplotlib.axes.Axes
    ) -> Callable:
        raise ValueError(self)

    @classmethod
    def new(
        cls, 
        x: Optional[ArrayOrCol]=None,
        y: Optional[ArrayOrCols]=None,
        x2: Optional[ArrayOrCol]=None,
        y2: Optional[ArrayOrCols]=None,
        c: Optional[ArrayOrCols]=None,
        colors: Optional[Union[Color, Colors]]=None,
        data: Optional[str] = None,
        kwargs: Optional[list[dict]] = None,
        **shared
    ):
        shape = cols_shape(y=y, y2=y2)
        assert shape is not None, shape
        if isinstance(colors, Colors) and c is None:
            c = list(np.linspace(
                0,
                1,
                shape[-1]
            ))
        elif c is not None:
            assert colors is not None, colors
        assert x is None or x2 is None, (x, x2)
        assert y is None or y2 is None, (y, y2)
        return cls(
            figure=None,
            axis=None,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            c=c,
            colors=colors,
            data=data,
            kwargs=kwargs,
            shared=shared,
        )

    def apply(
        self,
        axis: AxisRef,
        grid: Grid,
    ):
        if self.x is None and self.y is None:
            assert self.x2 is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_cols(
                self.y2, data=self.data
            )
            self_y = self.y2
        elif self.x is None:
            assert self.x2 is not None, self
            assert self.y is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert x_twin and not y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_cols(
                self.y, data=self.data
            )
            self_y = self.y
        elif self.y is None:
            assert self.x is not None, self
            assert self.y2 is not None, self
            assert axis.twinned, self
            assert isinstance(axis.obj, TwinnedAxes), axis
            ax, x_twin, y_twin = axis.obj
            assert y_twin and not x_twin, axis
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_cols(
                self.y2, data=self.data
            )
            self_y = self.y2
        else:
            assert isinstance(axis.obj, matplotlib.axes.Axes)
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_cols(
                self.y, data=self.data
            )
            ax = axis.obj
            self_y = self.y
        
        try:
            all_nan = np.all([
                np.isnan(yy) for yy in y
            ], axis = 0)
        except:
            raise ValueError([yy.shape for yy in y])

        assert len(all_nan) == len(x), dict(
            all_nan=len(all_nan),
            x=len(x),
            x_tail3= x[-3:],
            y = [yy.shape for yy in y]
        )
        not_nan = np.logical_not(all_nan)

        if isinstance(x, list):
            x = np.array(x)

        # x = x[not_nan]
        # y = [yy[not_nan] for yy in y]

        for xx in x:
            if xx != np.NAN:
                if isinstance(xx, dt.date):
                    date_format_x(ax, x, xx)
                break

        kwargs = self.kwargs
        if kwargs is None:
            kwargs = [{} for _ in y]

        assert len(kwargs) == len(y), dict(
            kwargs=kwargs,
            y=len(y)
        )
        color = None

        if isinstance(self.colors, Color):
            color = self.colors.color()
        if self.c is None:
            assert not isinstance(
                self.colors, Colors
            ), self.colors
            for i_label, (yy, kw) in enumerate(zip(y, kwargs)):
                label = (
                    kw.pop("label", i_label)
                    if not isinstance(
                        self_y[i_label], str
                    )
                    else self_y[i_label]
                )
                self.plot(ax)(
                    x,
                    yy,
                    color=color,
                    label=label,
                    **self.shared,
                    **kw
                )
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                prop=dict(size=6)
            )
            return
        assert isinstance(self.colors, Colors), self
        c = grid.unpack_cols(
            self.c, data=self.data, axis=x,
        )
        # TODO: fill boundaries with value range
        # if not given
        boundaries = self.colors.boundaries()
        bins = np.digitize(
            c,
            boundaries
        )
        boundaries = np.concatenate((
            np.zeros(1), boundaries
        ))
        boundaried = boundaries[bins]
        # each c is a time series (for instance)
        inds = np.linspace(0, len(c[0]) - 1, len(c[0]))
        diffs = boundaried[:, 1:] != boundaried[:,:-1]
        inds = np.stack([
            inds
            for _ in c
        ])
        cuts = np.where(
            diffs,
            inds[:,1:],
            inds[:,1:] * np.nan
        )
        segs = []
        for r, cu in zip(inds, cuts):
            cu = cu[~np.isnan(cu)]
            if not len(c):
                segs.append([r])
            try:
                segs.append(np.split(
                    r,
                    cu,
                    axis=0
                    # + 1 ?
                ))
            except:
                raise ValueError(r, c)
        plot = self.plot(ax)
        for label, (yy, ii, cc, kw) in enumerate(zip(
            y, segs, c, kwargs
        )):
            label = (
                kw.pop("label", label)
                if not isinstance(
                    self_y[label], str
                )
                else self_y[label]
            )
            for iii in ii:
                iii = np.asarray(iii, dtype=int)
                col = self.colors.sample(
                    cc[iii][0]
                ).rgba
                plot(
                    x[iii],
                    yy[iii],
                    color=col,
                    label=label,
                    **kw,
                )
        # box = ax.get_position()
        # ax.set_position([
        #     box.x0,
        #     box.y0,
        #     box.width * 0.8,
        #     box.height
        # ])
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            prop=dict(size=6)
        )

# -----------------

class Scatters(Continuous_2D):
    
    def plot(self, axes: matplotlib.axes.Axes):
        return axes.scatter

class Lines(Continuous_2D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.plot

# -----------------

# TODO: 

# grid lines

# zero line

# axis frequency (eg. 1y, 10y, etc.)

# bold vs not?

# market labels & legend

# axis labels

# null split line (so a line, but where nulls show as zero rather than joined - ie. split into segments and plot separately)

# -----------------

# grid = Grid.new().with_data(
#     df_a=pd.DataFrame({
#         "a": [0, 1, 2]
#     })
# ).with_axes(
#     [["a"]], twin_x = False, twin_y=True,
# ).with_chart(
#     fig.axis.a,
#     Line.new(x="a", y=[1, 2, 3], data="df_a"),
# ).with_chart(
#     fig.axis.a.twin,
#     Scatter.new(x=[0, 1, 2], y2=[2, 4, 6])
# ).with_chart(
#     fig.axis.a.twin,
#     Line.new(x=[1, 2, 3], y2=[1, 2, 3])
# ).with_chart(
#     fig.axis.a.twin,
#     Scatter.new(x=[1, 2, 3], y2=[1, 2, 3])
# )
# print(grid.charts)
# grid.show()


# TODO: mutable grid? that they can all share references to? as we edit the grid - or you create all the axes etc. up front, and then we don't worry about acc the object?