from typing import Any
from frozendict import frozendict

from . import nav
from . import page
from . import model

from .examples import pivot

Shared = frozendict[str, Any]
