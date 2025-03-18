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
    scope: Scope.Aliases | None


class Aliased(Aliased_Kw, Plugin):

    def alias(
        self,
        alias: str | None,
        **kwargs
    ):
        scope = self.scope
        if scope is None:
            scope = Scope.Aliases.new(alias=alias, kwargs = frozendict(kwargs))
        elif alias is not None:
            scope = scope._replace(alias=alias, kwargs = frozendict(kwargs))
        return self._replace(scope=scope)


#  ------------------
