
import os
import sys
sys.path.append("./src")

from typing import Callable

import ceg

fut = ceg.objs.dev.fut
exp = ceg.app.explorer

os.environ["FRD_DATA"] = "./__local__/frd"

if __name__ == "__main__":
    app = exp.prepare_app(
        "dev",
        "./__local__/cache",
        ceg.app.dev.domains.DOMAINS
    ).run()