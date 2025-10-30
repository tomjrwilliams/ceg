
import sys
from pathlib import Path

sys.path.append("./src")

import ceg
sims = ceg.app.dev.academic.sims

SIMS = {
    "order_driven": sims.order_driven.run,
    #
}

def count_trials(dp: Path) -> int:
    res = 0
    for f in dp.iterdir():
        if f.name.startswith("trial"):
            res += 1
    return res

def parse_params(sim: str, *params: str):
    res: dict[str, str | None] = {
        "dir": "./__local__",
        "variant": None,
        "trial": None,
    }
    for i, v in enumerate(params):
        if "=" in v:
            k, v = v.split("=")
        elif i == 0:
            k = "variant"
        elif i == 1:
            k = "trial"
        else:
            raise ValueError(params)        
        res[k] = v

    variant = res.pop("variant")
    trial = res.pop("trial")

    dir = res.pop("dir")
    assert dir is not None

    dp = Path(dir) / sim

    if variant is not None:
        dp = dp / variant       

    dp.mkdir(exist_ok=True, parents=True) 

    if trial is None:
        trial = count_trials(dp)
    
    dp = dp / f"trial-{str(trial).rjust(3, '0')}"
    dp.mkdir(exist_ok=True, parents=True) 
    
    fp = dp / "results.npz"
    
    return {
        **dict(fp=fp, variant=variant), **res
    }

if __name__ == "__main__":
    args = sys.argv

    sim = args[1]
    params = args[2:]

    SIMS[sim](**parse_params(sim, *params))
