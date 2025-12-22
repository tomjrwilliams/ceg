

import os, sys, signal
sys.path.append("./src")

from typing import Callable
from pathlib import Path

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
    
    if len(project_name) == 0:
        exit()

    os.environ["TEST_LOGS"] = os.environ.get(
        "TEST_LOGS", f"./__local__/logs/{project_name}.log"
    )
    log_fp = Path(os.environ["TEST_LOGS"])
    
    if log_fp.exists():
        try:
            os.remove(log_fp)
        except:
            pass
    log_fp.parent.mkdir(parents=True, exist_ok=True)

    import ceg

    project = ceg.app.dev.PROJECTS[project_name]()
    
    if len(args) > 1:
        command = args[1]
        command_args = args[2:]
        if command == "reset":
            project.reset()
        if len(command_args):
            if command_args[0] == "die":
                os.kill(os.getpid(), signal.SIGINT)

    app = ceg.objs.vis.explorer.prepare_app(
        app_name,
        app_cache,
        project.domains
    ).run()

# py -m streamlit run ./src/ceg/app/dev/run.py p001 reset