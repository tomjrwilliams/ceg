
from typing import NamedTuple
from frozendict import frozendict

from ..graphs import Scope

from ..graphs import *
from ..types import *

from .. import Node
from .. import Ref

#  ------------------

class Alias_Kw(NamedTuple):
    alias: str | None
    aliases: frozendict[Ref.Any, str]

class Alias(Alias_Kw, Scope):

    @classmethod
    def new(
        cls,
        alias: str | None = None,
        aliases: frozendict[Ref.Any, str] = frozendict() # type: ignore
    ):
        return cls(alias, aliases)
    
    def merge(
        self,
        node: Node.Any,
        ref: Ref.Any,
        scope: Scope | None,
    ) -> Scope:
        if scope is None:
            return self._replace(
                aliases=self.aliases.set(ref, self.alias),
                alias=None
            )
        assert isinstance(scope, Alias), scope
        return self._replace(
            aliases = (
                self.aliases | scope.aliases
            ).set(ref, self.alias),
            alias=None,
        )

    def contains(
        self, graph: Graph, node: Node.Any, event: Event
    ) -> bool:
        return event.ref in self.aliases

#  ------------------
