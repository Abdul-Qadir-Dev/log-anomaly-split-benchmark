"""Step 13: confirm the newly-seeded deeplog arm is bit-reproducible.
Trains and scores deeplog twice, in two separate processes, on the selftest
HDFS_v1/random/seed0 split, and checks the resulting AUPRC/AUROC match to 6
decimal places. Writes results/audit/deeplog_determinism_check.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ONE_RUN = """
import sys
sys.path.insert(0, "src")
import pandas as pd
from arms.deeplog import DeepLogArm
from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell

dataset, split, seed = "HDFS_v1", "random", 0
sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", True))
d = __import__("pathlib").Path("results/selftest") / dataset / split / f"seed{seed}"
ids = {}
for name in ["train", "val", "test"]:
    with open(d / f"{name}_ids.json") as f:
        ids[name] = json.load(f)

def subset(ids_):
    return sessions[sessions["session_id"].isin(set(ids_))].reset_index(drop=True)

train = subset(ids["train"])
val = subset(ids["val"])
test = subset(ids["test"])
train = train[train["label"] == 0].reset_index(drop=True)

n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", True)))
ctx = ArmContext(dataset=dataset, selftest=True, n_vocab=n_vocab, seed=seed)
arm = DeepLogArm()
arm.fit(ctx, train, val)
score_val = arm.score(ctx, val)
score_test = arm.score(ctx, test)
m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
print(json.dumps({"auroc": m["auroc"], "auprc": m["auprc"], "f1_val": m["f1_val"], "f1_oracle": m["f1_oracle"]}))
"""


def run_once():
    out = subprocess.check_output([sys.executable, "-c", "import json\n" + ONE_RUN], text=True)
    return json.loads(out.strip().splitlines()[-1])


def main():
    r1 = run_once()
    r2 = run_once()
    match = {k: round(r1[k], 6) == round(r2[k], 6) for k in r1}
    result = {"run1": r1, "run2": r2, "match_to_6dp": match, "all_match": all(match.values())}
    Path("results/audit").mkdir(parents=True, exist_ok=True)
    with open("results/audit/deeplog_determinism_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result["all_match"]:
        raise SystemExit("determinism check FAILED")


if __name__ == "__main__":
    main()
