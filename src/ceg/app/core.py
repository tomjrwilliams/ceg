
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Type,
)
from dataclasses import replace

import argparse
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import os
import shutil
os.environ["FRD_DATA"] = "./__local__/data/frd"

from ..engine import *

@dataclass(frozen=True)
class App:
    root: str
    build: Callable[[argparse.Namespace], tuple[frozendict, AppConfig]]
    args: t.frozen.dict[str, frozendict] = frozendict(
        app=frozendict(type=str)
    )

    def with_args(self, **args: frozendict | dict):
        return replace(self, args = (
            self.args
            | frozendict({
                k: frozendict(kw) for k, kw in args.items()
            })
        ))

    def run(self):
        ROOT = self.root

        parser = argparse.ArgumentParser()

        for k, kw in self.args.items():
            parser.add_argument(k, **kw)

        args = parser.parse_args()
        acc, config = self.build(args)

        return config.run(
            self.root,
            acc,
        )


@dataclass(frozen=True)
class AppConfig:
    graph: Type[Graph]
    projects: tuple[str, ...]
    configs: t.frozen.dict[str, Bind |Apply] = frozendict()
    
    def with_configs(self, **configs: Bind | Apply):
        res = self.configs
        for k, v in configs.items():
            if not isinstance(v, (Bind, Apply)):
                raise ValueError(v, k)
            res = res.set(k, v)
        return replace(self, configs = res)
    

    def run(
        self,
        ROOT: str,
        res: frozendict,
    ):
        from redis import Redis

        conn = Redis("localhost", port =7777)

        for env in ["prod"]:
            for table in [
                "Events",
                "Objects",
                "Snaps",
                "Deps",
                "ValuesJSON",
                "ValuesDF",
                "ValuesSeries",
            ]:
                # ChunksPlSeriesF64/cv-ETF-GLD-comb-0-gross/1
                # etc.
                # keys = conn.keys(f"{env}/{table}/*")
                keys = conn.keys(f"{env}/{table}/{ROOT}-*")
                if len(keys):
                    print("DELETE", keys)
                    conn.delete(*keys)
        
        c = Collection(ROOT)

        with context.graph(ROOT, t=self.graph):
            projects = {
                k: Project.new(k) for k in self.projects
            }
            for k, v in projects.items():
                context.get().with_obj(v)
                c = c.with_obj(v, k)

            list(projects.values())[0].set_storage_defaults([
                # TODO: parametrise out these as well as proejcts
                "config",
                "eq",
                "fut",
                "fx",
                # --
            ] + list(self.projects))

            for k, config in self.configs.items():
                if not isinstance(config, (Apply)):
                    try:
                        c = config(c, config=k)
                    except:
                        raise ValueError(c, self.configs)

            context.get().sync()

            for k, config in self.configs.items():
                if isinstance(config, Apply):
                    res = config(c, res, config=k)

APPS: dict[str, App] = {

}