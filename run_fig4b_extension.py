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

# Denser sampling around the reported critical region
train_nums = [18, 20, 22, 24, 26, 28, 30]

# More seeds than the original experiment
seeds = list(range(20))

steps = 10_000
threshold_rqi = 0.95

rows = []

for train_num in train_nums:
    for seed in seeds:

        print(
            f"\ntrain_num={train_num}/55 "
            f"({train_num/55:.3f}), seed={seed}"
        )

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

        # Save continuously
        pd.DataFrame(rows).to_csv(
            RESULTS / "fig4b_extension.csv",
            index=False
        )

        print(row)

print("\nFinished.")