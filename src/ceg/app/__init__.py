
from typing import Any
from frozendict import frozendict

from . import nav
from . import page
from . import model

Shared = frozendict[
    str, Any
]