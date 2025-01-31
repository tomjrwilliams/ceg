
from typing import NamedTuple
from frozendict import frozendict

from ..graphs import Plugin

from ..graphs import *
from ..types import *

from .. import Node
from .. import Ref

from .. import Scope

#  ------------------

class Aliased_Kw(NamedTuple):
    scope: Scope.Alias | None

class Aliased(Aliased_Kw, Plugin):
    
    def alias(
        self, alias: str | None, **aliases: Ref.Any,
    ):
        scope = self.scope
        if scope is None:
            scope = Scope.Alias.new(alias=alias)
        elif alias is not None:
            scope = scope._replace(alias=alias)
        if len(aliases):
            scope = scope._replace(
                aliases=scope.aliases | {
                    r: s for s, r in aliases.items()
                }
            )
        return self._replace(scope=scope)
        
#  ------------------
