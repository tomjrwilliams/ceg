from __future__ import annotations
from typing import Optional, NamedTuple, Union, Callable

from frozendict import frozendict

from functools import lru_cache

import numpy as np
import pandas as pd

import matplotlib.figure
import matplotlib.axis
import matplotlib.axes

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
    def obj(self) -> matplotlib.axes.Axes:
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
            return axes[self.axis]
        assert fig.figure is None, (
            grid.figures,
            fig.figure
        )
        if self.twinned:
            return grid.twins[self.axis]
        return grid.axes[self.axis]

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
        return grid.figures[self.figure]
    
    @property
    def axes(self) -> frozendict[
        str, matplotlib.axes.Axes
    ]:
        assert self.grid is not None, self
        grid = self.grid
        if self.figure is None:
            assert grid.figures is None, (
                grid.figures, self.figure
            )
            # assert no nested?
            return grid.axes
        return grid.axes[self.figure]

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

class Grid(NamedTuple):

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
            axes=frozendict(),
            twins=frozendict(),
            charts=(),
            dfs=frozendict()
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
            ncols=n_cols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        n_cols = n_cols[0]
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
            labels,
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
            axis.set_title(label)
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
        axis = axis.with_grid(self)
        chart = chart._replace(
            figure=axis.fig.figure,
            axis=axis.axis,
        )
        chart.apply(axis, self)
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
        axis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(col, (int, float)):
            assert axis is not None, (col, axis)
            return [col for _ in axis]
        if isinstance(col, str):
            assert data is not None, (col, data)
            return self.dfs[data][col]
        assert isinstance(col, (list, np.ndarray)), col
        return col
    
    def unpack_cols(
        self: Grid,
        cols: ArrayOrCols,
        data: Optional[str] = None,
        axis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(cols, np.ndarray):
            return [
                cols[:, i]
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
) -> tuple[int, ...]:
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
    ],
    n: int
):
    cmap = cmap.resampled(n)
    colors: np.ndarray = cmap.__dict__["colors"]
    rgba = map(tuple(colors.tolist()))
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
        l, r = self.range()
        width = (r - l) / n
        return np.array([
            l + width + (i * width)
            for i in range(n)
        ])

    def sample(
        self, cval: Union[int, float]
    ) -> tuple[float, float, float, float]:
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
        return Color(rgba=rgba).with_hex()

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
        return self.with_hex().hex

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
    return Color(name=name)

def hex(s: str) -> Color:
    # TODO: with rgba
    return Color(hex=s)

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
    return Color(rgba=rgba).with_hex()

# -----------------

class Mark_Kw(NamedTuple):
    figure: Optional[str]
    axis: Optional[str]
    
class Mark(Mark_Kw):
    
    def apply(
        self,
        axis: matplotlib.axes.Axes,
        grid: Grid,
    ):
        raise ValueError(self)

# -----------------

class Mark_2D_Kw(NamedTuple):
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

class Mark_2D(Mark_2D_Kw, Mark):

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
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert x_twin and y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        elif self.x is None:
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert x_twin and not y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_col(
                self.y, data=self.data
            )
        elif self.y is None:
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert y_twin and not x_twin, axis
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_col(
                self.y2, data=self.data
            )
        else:
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

# -----------------

class Scatter(Mark_2D):
    
    def plot(self, axes: matplotlib.axes.Axes):
        return axes.scatter

class Line(Mark_2D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.plot

class Bar(Mark_2D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.bar
    
# -----------------

class Marks_2D_Kw(NamedTuple):
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


class Marks_2D(Marks_2D_Kw, Mark):

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
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert x_twin and y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_cols(
                self.y2, data=self.data
            )
        elif self.x is None:
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert x_twin and not y_twin, axis
            x = grid.unpack_col(
                self.x2, data=self.data
            )
            y = grid.unpack_cols(
                self.y, data=self.data
            )
        elif self.y is None:
            assert axis.twinned, self
            ax, x_twin, y_twin = axis.obj
            assert y_twin and not x_twin, axis
            x = grid.unpack_col(
                self.x, data=self.data
            )
            y = grid.unpack_cols(
                self.y2, data=self.data
            )
        else:
            ax = axis.obj
        
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
            for yy, kw in zip(y, kwargs):
                self.plot(ax)(
                    x,
                    yy,
                    color=color,
                    **self.shared,
                    **kw
                )
            return
        assert isinstance(self.colors, Colors), self
        c = grid.unpack_cols(
            self.c, data=self.data, axis=x,
        )
        # TODO: fill boundaries with value range
        # if not given
        boundaries = self.colors.boundaries()
        # maybe multi index work? maybe have to loop
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
        for yy, ii, cc, kw in zip(
            y, segs, c, kwargs
        ):
            plot(
                x[ii],
                y[ii],
                color=cc[ii][0],
                **self.kwargs,
            )

# -----------------

class Scatters(Marks_2D):
    
    def plot(self, axes: matplotlib.axes.Axes):
        return axes.scatter

class Lines(Marks_2D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.plot

class Bars(Marks_2D):

    def plot(self, axes: matplotlib.axes.Axes):
        return axes.bar

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