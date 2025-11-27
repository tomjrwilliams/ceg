

import os, sys, signal
sys.path.append("./src")

from typing import Callable
from pathlib import Path

import ceg

fut = ceg.objs.dev.fut

os.environ["FRD_DATA"] = "./__local__/frd"

project_name: str = ""
app_name: str = "dev"
app_cache: str = "./__local__/cache"

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args):
        app_name = project_name = args[0]
        # TODO: dev in path?
        app_cache = f"./__local__/apps/{app_name}"
        Path(app_cache).mkdir(exist_ok=True, parents=True)

    project = ceg.app.dev.PROJECTS[project_name]

    if len(args) > 1:
        command = args[1]
        if command == "reset":
            project.reset()
            os.kill(os.getpid(), signal.SIGINT)

    app = ceg.objs.vis.explorer.prepare_app(
        app_name,
        app_cache,
        project.domains
    ).run()