import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "toy"
RESULTS = ROOT / "reproduction" / "results"

sys.path.insert(0, str(TOY))

from train_add import train_add

RESULTS.mkdir(parents=True, exist_ok=True)

train_nums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 54]
seeds = list(range(20))

steps = 10_000
threshold_rqi = 0.95

OUTFILE = RESULTS / "fig4b_20seeds.csv"

rows = []

for train_num in train_nums:
    for seed in seeds:

        print(f"\nRunning train_num={train_num}, seed={seed}")

        start = time.perf_counter()

        d = train_add(
            train_num=train_num,
            seed=seed,
            steps=steps,
            eff_steps=1,
            log_freq=10_000,
        )

        runtime = time.perf_counter() - start

        final_rqi = float(d["rqi"][-1])
        iter_rqi = int(d["iter_rqi"])

        # train_add uses steps-1 when the threshold was never reached.
        reached_rqi = final_rqi >= threshold_rqi or iter_rqi < steps - 1

        row = {
            "train_num": train_num,
            "train_fraction": float(d["train_ratio"]),
            "seed": seed,
            "dof": int(d["dof"]),
            "iter_rqi": iter_rqi,
            "reached_rqi": reached_rqi,
            "final_rqi": final_rqi,
            "final_train_acc": float(d["acc_train"][-1]),
            "final_test_acc": float(d["acc_test"][-1]),
            "runtime_seconds": runtime,
        }

        rows.append(row)

        # Save after every run, so results survive interruptions.
        df = pd.DataFrame(rows)
        df.to_csv(OUTFILE, index=False)

        print(row)

print("\nFinished all runs.")
print(pd.DataFrame(rows))