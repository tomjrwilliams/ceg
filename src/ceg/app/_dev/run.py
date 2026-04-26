import os, sys, signal

sys.path.append("./src")

import json
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

    streamlit = ceg.apis.st.in_streamlit()

    logger, log = ceg.apis.logs.logger(
        __file__, streams=dict(formatter=True)
    )
    PROJECTS = {}
    Project = ceg.app._dev.PROJECTS[project_name]

    os.environ["STREAMLIT"] = str(streamlit)

    args_fp = Path("./src/ceg/app/dev/ARGS.json")
    if args_fp.exists():
        with args_fp.open("r") as f:
            args = json.load(f)
    else:
        args = {}

    command = args.get("command", None)

    if command == "reset":
        log("INFO")(command="reset")
        ceg.vis.explorer.load_page(
            None, Project, reset=True
        )

        if streamlit:
            log("INFO")(clearing_cache="streamlit")
            ceg.apis.st.library().cache_data.clear()

    if "die" in args:
        log("INFO")(command="die")
        os.kill(os.getpid(), signal.SIGINT)

    if streamlit:
        log("INFO")(running_app=app_name)
        app = ceg.vis.explorer.prepare_app(
            app_name,
            app_cache,
            Project,
            suffix_root=False,
        ).run()

# py -m streamlit run ./src/ceg/app/dev/run.py p001
