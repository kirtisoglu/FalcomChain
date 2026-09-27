from .errors import *
from .snapshot import Recorder
from .tree import *

# Public API of the subpackage: everything star-imported above, but NOT the
# submodule names (``tree``, ``errors``, ``snapshot``). Without this list a
# star-import of this package (``falcomchain/__init__.py`` does one) would
# also copy the submodule attribute ``tree`` into ``falcomchain``, shadowing
# the ``falcomchain.tree`` subpackage and breaking
# ``import falcomchain.tree.tree as ...``.
__all__ = [
    name for name in dir()
    if not name.startswith("_") and name not in ("tree", "errors", "snapshot")
]
