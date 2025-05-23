import re
from typing import NamedTuple
from frozendict import frozendict

from ..graphs import Scope

from ..graphs import *
from ..types import *

from .. import Node
from .. import Ref

#  ------------------


class Aliases_Kw(NamedTuple):
    alias: str | None
    kwargs: frozendict | None
    aliases: frozendict[tuple[Ref.Any, str], frozendict]


class Aliases(Aliases_Kw, Scope):

    @classmethod
    def new(
        cls,
        alias: str | None = None,
        kwargs: frozendict | None = None,
        aliases: frozendict[tuple[Ref.Any, str], frozendict] = frozendict(), # tpyE: ignore
    ):
        return cls(alias, kwargs, aliases)

    def merge(
        self,
        node: Node.Any,
        ref: Ref.Any,
        scope: Scope | None,
    ) -> Scope:
        if scope is None and self.alias is None:
            return self
        elif scope is None:
            return self._replace(
                alias=None,
                kwargs=None,
                aliases=self.aliases.set(
                    (ref, self.alias), self.kwargs
                )
            )
        assert isinstance(scope, Aliases), scope
        if self.alias is not None:
            self = self._replace(
                alias=None,
                kwargs=None,
                aliases=self.aliases.set(
                    (ref, self.alias), self.kwargs
                )
            )
        return self._replace(
            alias=None,
            kwargs=None,
            aliases=(scope.aliases | self.aliases)
        )

    def contains(
        self, graph: Graph, node: Node.Any, event: Event
    ) -> bool:
        # TODO: sensibly cache
        for ref, _ in self.aliases.keys():
            if event.ref == ref:
                return True
        return False

#  ------------------
