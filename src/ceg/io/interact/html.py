
from __future__ import annotations
from typing import Optional, NamedTuple, Union, Callable, cast, Protocol

from pathlib import Path
from contextlib import contextmanager

from frozendict import frozendict

from functools import lru_cache

import numpy as np
import pandas as pd

def page_header(title):
    return f"""
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{title}</title>
<style type="text/css">""" + r"""
html *
{
   font-family: Arial !important;
}
html, body, #header {
    margin: 0 !important;
    padding: 0 !important;
}
body {
   margin: 0;
   padding: 0;
}
</style>
</head>
"""

class Identifier(NamedTuple):
    name: str
    parent: Optional[Identifier]
    page: str | None

    def __repr__(self):
        return self.uid

    @classmethod
    def new(cls, name: str):
        return cls(name=name, parent=None, page=None)
    
    @property
    def uid(self) -> str:
        if self.parent is None:
            return self.name
        return self.parent.uid + "." + self.name

    def fp(self, extension: str, relative: bool = False) -> Path:
        assert self.page is not None, self
        if relative:
            fp = Path(f"./{self.uid}.{extension}")
        else:
            fp = Path(self.page) / f"./{self.uid}.{extension}"
            fp = fp.resolve()
        return fp


class Div(NamedTuple):
    id: Identifier

    def html(self) -> tuple[str, str]:
        raise ValueError(self)

    def on_add(self, div: Div) -> Div:
        return self

    def add(
        self, div: Div, page: Page
    ):
        div = div._replace(id = div.id._replace(
            parent=self.id, page = page.url
        ))
        self = self.on_add(div)
        page = page.set(self)
        page = page.set(div)
        return page, self, div

    @contextmanager
    def open(self, extension: str, mode = "w+"):
        assert self.id.page is not None, self.id
        page = Path(self.id.page)
        page.mkdir(parents=True, exist_ok=True)
        fp = self.id.fp(extension)
        with fp.open(mode) as f:
            yield f, fp

import shutil

# TODO: possibly a better approach is 
# quickly run and dump a sim

# given a cache path
# check the graph matches on re run, if not, re run

# could also auto generate (optionally) by hash of the graph string repr, delete over the last n most recent auto generated 

# and then analytics to make reading out easy?
# problem i guess is de-serialising the whole graph is a pain

class Page(NamedTuple):
    url: str
    title: str
    divs: frozendict[Identifier, Div]
    root: Optional[Identifier]

    @classmethod
    def new(
        cls,
        url: str, 
        title: str,
    ) -> Page:
        assert "__local__" in url, url # TODO: other allowed dirs, so we don't accidentally delte root or whatever
        Path(url).mkdir(exist_ok=True, parents=True)
        shutil.rmtree(url)
        Path(url).mkdir(exist_ok=True, parents=True)
        return cls(
            url=url,
            title=title,
            divs=frozendict(), # type: ignore
            root=None,
        )

    @contextmanager
    def open(self, mode = "w+"):
        page = Path(self.url)
        page.mkdir(exist_ok=True,parents=True)
        with (page / "index.html").open(mode) as f:
            yield f, self.url

    def write(self):
        html = self.html()
        with self.open(mode="w+") as (f, fp):
            f.write(html)

    def get(self, id: Identifier) -> Div | None:
        return self.divs.get(id)

    def add(self, div: Div, parent: Div | None = None):
        if parent is None:
            div = div._replace(id = div.id._replace(
                page = self.url
            ))
            self = self.set(div)
            return self, div
        self, parent, div = parent.add(div, self)
        return self, parent, div

    def set(self, div: Div):
        if div.id.parent is None:
            if self.root != div.id:
                assert not len(self.divs), self
                self = self._replace(root=div.id)
        return self._replace(
            divs=self.divs.set(div.id, div)
        )

    def html(self) -> str:
        seen: set[Identifier] = set()
        s = page_header(self.title)
        assert self.root is not None, self
        stack: list[Identifier | str]= [self.root]
        while len(stack):
            *stack, todo = stack
            if isinstance(todo, str):
                s += todo
                continue
            assert isinstance(todo, Identifier)
            id = todo
            assert id not in seen, (seen, id)
            seen.add(id)
            div = self.get(id)
            assert div is not None, (self, id)
            l, r = div.html()
            s += l
            stack.append(r)
            children = [
                child_id
                for child_id, child_div
                in self.divs.items()
                if child_id.parent == id
            ]
            stack.extend(reversed(children))
        return s

class VerticalContainerKW(NamedTuple):
    id: Identifier
    width: int | float | None #  none assume equal share of parent
    title: str | None
    style: str | None
    divs: tuple[Identifier, ...]

class VerticalContainer(VerticalContainerKW, Div):

    def html(self) -> tuple[str, str]:
        s = "<div"

        s = add_width(s, self)
        # TODO: height? or leave unbounded to grow (inside fixed moasic)

        s = add_style(s, self, default="display:flex;justify-content:center;align-items:center;flex-grow: 1;flex-basis: 0;flex-direction:column;")

        s += ">"

        if self.title:
            t = f'<h2 style="display:flex;justify-content:center;align-text:center;width:100%;margin:0;">{self.title}</h2>'
            s += t

        return s, "</div>"

    def on_add(self, div: Div):
        return self._replace(
            divs = self.divs + (div.id,)
        )
    
    @classmethod
    def new(
        cls,
        page: Page,
        parent: Div | None = None,
        name: str | None=None,
        width: int | float | None=None,
        title: str | None=None,
        style: str | None = None,
    ):
        if name is None:
            name = str(count_children(page, parent))
        id = Identifier.new(name)
        div = VerticalContainer(
            id, width=width, title=title,style=style, divs=tuple()
        )
        return page.add(div, parent=parent)

Col = Column = VerticalContainer

class VerticalMosaicKW(NamedTuple):
    id: Identifier
    width: int | float | None
    title: str | None
    style: str | None
    containers: tuple[Identifier, ...]

class VerticalMosaic(VerticalMosaicKW, Div):

    def html(self) -> tuple[str, str]:
        s = "<div"

        s = add_width(s, self, default = "100%")
        # TODO: height? or leave unbounded to grow (inside fixed moasic)

        s = add_style(s, self, default="display:flex;align-items:start;justify-content:center;overflow-y:scroll;height:100%;overflow-x:hidden;")

        s += ">"

        if self.title:
            t = f'<h1 style="display:flex;justify-content:center;align-text:center;width:100%;margin:0;">{self.title}</h1>'
            s += t

        return s, "</div>"

    def on_add(self, div: Div):
        assert isinstance(div, (
            VerticalContainer,
            # TODO: other containers
        ))
        return self._replace(
            containers = self.containers + (div.id,)
        )

    @classmethod
    def new(
        cls,
        page: Page,
        parent: Div | None = None,
        name: str | None=None,
        width: int | float | None=None,
        title: str | None=None,
        style: str | None = None,
    ):
        if name is None:
            name = str(count_children(page, parent))
        id = Identifier.new(name)
        div = VerticalMosaic(
            id, width=width, title=title, style=style, containers=tuple()
        )
        return page.add(div, parent=parent)

Mosaic = VerticalMosaic

class ImageKW(NamedTuple):
    id: Identifier
    extension: str
    height: int | float | None
    width: int | float | None #  none assume equal share of parent
    style: str | None
    title: str | None

class Image(ImageKW, Div):
    
    def html(self):
        fp = self.id.fp(self.extension, relative=True)
        
        s = ""
        if self.title:
            t = f'<h3 style="display:flex;justify-content:center;align-text:center;width:100%;margin:0;">{self.title}</h3>'
            s += t

        s += f'<img src="{str(fp)}"'
        if self.title is not None:
            s += f' alt="{self.title}"'

        s = add_height(s, self)
        s = add_width(s, self)

        s = add_style(s, self, default="max-width:100%;height:auto;")
        # default = "vertical-align:middle;margin-left:auto;margin-right:auto;" (assume flex flow)

        s += ">"

        return s, ""


    @classmethod
    def new(
        cls,
        page: Page,
        parent: VerticalContainer, # or horizontal etc.?
        name: str | None=None,
        extension: str = "png",
        height: int | float | None=None,
        width: int | float | None=None,
        style: str | None = None,
        title: str | None=None,
    ):
        if name is None:
            name = str(len(parent.divs))
        id = Identifier.new(name)
        div = Image(
            id, extension=extension, width=width, height=height, title=title, style=style,
        )
        return page.add(div, parent=parent)


class HasHeight(Protocol):
    height: int | float | None

def add_height(s: str, div: HasHeight, default: str | None = None) -> str:
    height = div.height or default
    if height is None:
        return s
    return s + f' height="{height}"'

class HasWidth(Protocol):
    width: int | float | None

def add_width(s: str, div: HasWidth, default: str | None = None) -> str:
    width = div.width or default
    if width is None:
        return s
    return s + f' width="{width}"'

class HasStyle(Protocol):
    style: str | None

def add_style(
    s: str,
    div: HasStyle,
    default: str | None = None,
) -> str:
    style = div.style or default
    if style is None:
        return s
    return s + f' style="{style}"'

def count_children(page: Page, parent: Div | None = None):
    if parent is None:
        assert len(page.divs) == 0, page
        return 0
    assert parent is not None, (parent)
    return len([
        id for id in page.divs.keys()
        if id.parent == parent.id
    ])