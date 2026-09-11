"""Private judge: scores a candidate on the 2025 holdout and returns ONLY an aggregate result.

The holdout labels live outside the repository (LEAGUE_PRIVATE_LABELS env var, or
--labels path). Candidate code never receives 2025 actuals except as training rows for weeks
strictly before the test week (true expanding walk-forward, same as production).
Budget: at most MAX_PER_ROUND private submissions per round (league/registry/private_log.jsonl).
Output (and the only thing written to the shared registry):
  {"submission_id", "label", "round", "private_score", "rank", "eligible_for_promotion", "reason_code"}
Usage: python league/judge/private_judge.py --candidate league/candidates/exp_001 --round 1 [--seed 17]
       python league/judge/private_judge.py --baseline ffa|model_burke --round 0   (reference runs, not budgeted)
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as J
LG = os.path.join(ROOT, "league")
MAX_PER_ROUND = 2
GATES = {"min_private_relative_gain": 0.002, "min_visible_relative_gain": 0.005, "max_worse_seasons": 0}

def load_holdout(labels_path):
    feat = pd.read_parquet(os.path.join(LG, "data", "holdout_features.parquet"))
    lab = pd.read_parquet(labels_path)
    return feat.merge(lab, on=["player_id", "season", "week"], how="left")

def private_log():
    p = os.path.join(LG, "registry", "private_log.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate"); ap.add_argument("--baseline"); ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17); ap.add_argument("--labels", default=os.environ.get("LEAGUE_PRIVATE_LABELS"))
    A = ap.parse_args()
    if not A.labels or not os.path.exists(A.labels): raise SystemExit("private labels path missing (LEAGUE_PRIVATE_LABELS)")
    if os.path.commonpath([os.path.abspath(A.labels), ROOT]) == ROOT: raise SystemExit("labels must live outside the repository")
    vis = J.load_frame(); hold = load_holdout(A.labels)
    full = pd.concat([vis, hold], ignore_index=True); full["t"] = full.season * 100 + full.week
    log = private_log()
    label = A.baseline or os.path.basename(A.candidate.rstrip("/\\"))
    if A.candidate:
        used = [r for r in log if r.get("round") == A.round and r.get("kind") == "candidate"]
        if len(used) >= MAX_PER_ROUND: raise SystemExit(f"private budget exhausted for round {A.round} ({MAX_PER_ROUND})")
        hits, h = J.static_check(os.path.join(A.candidate, "model.py"))
        if hits: raise SystemExit(f"banned I/O in model.py: {hits}")
        model = J.load_candidate(A.candidate)
        t0 = time.time(); p, secs = J.walk_forward(model, full, [2025], A.seed, log=lambda *a: None)
    else:
        sys.path.insert(0, os.path.join(LG, "baselines")); import run_baselines as B
        p = B.baseline_predictions(A.baseline, full, [2025], A.seed); secs = 0
    ev = J.eval_rows(full, [2025])
    # reference: FFA on the holdout (computed here, never exposed beyond the aggregate)
    ffa_p = B_ffa = None
    sys.path.insert(0, os.path.join(LG, "baselines")); import run_baselines as B
    ffa_p = B.baseline_predictions("ffa", full, [2025], A.seed)
    ffa_m = J.metrics(ev, ffa_p)
    inc_pin = next((r for r in reversed(log) if r.get("label") == "model_burke_prod"), {}).get("_pinball")
    if A.baseline == "model_burke_prod": inc_pin = J.metrics(ev, p)["pinball"]      # the incumbent references its own spread
    m = J.metrics(ev, p, ref_pinball=inc_pin, ffa=ffa_m)
    # incumbent private score for the gate
    inc = next((r for r in reversed(log) if r.get("label") == "model_burke_prod" and r.get("kind") == "baseline"), None)
    score = m.get("composite", float("nan"))
    if A.candidate:
        vis_regs = [r for r in J.load_registry() if r.get("label") == label and r.get("kind") == "visible" and r.get("valid")]
        vis_inc = [r for r in J.load_registry() if r.get("label") == "model_burke_prod" and r.get("kind") == "visible"]
        reason = "OK"; eligible = True
        if not vis_regs: reason, eligible = "NO_VALID_VISIBLE_SUBMISSION", False
        elif inc is None: reason, eligible = "NO_INCUMBENT_PRIVATE_REFERENCE", False
        else:
            vg = vis_regs[-1]["metrics"]["composite"] / vis_inc[-1]["metrics"]["composite"] - 1 if vis_inc else 0
            pg = score / inc["private_score"] - 1
            worse = sum(1 for s, x in vis_regs[-1]["metrics"]["by_season"].items() if x["mae"] > vis_inc[-1]["metrics"]["by_season"][str(s)]["mae"] * 1.01) if vis_inc else 0
            if vg < GATES["min_visible_relative_gain"]: reason, eligible = "VISIBLE_GAIN_TOO_SMALL", False
            elif pg < GATES["min_private_relative_gain"]: reason, eligible = "PRIVATE_GAIN_TOO_SMALL", False
            elif worse > GATES["max_worse_seasons"]: reason, eligible = "WORSE_ON_A_SEASON", False
            elif m["coverage"] < J.CONFIG["min_coverage"]: reason, eligible = "INCOMPLETE_PREDICTIONS", False
    else:
        reason, eligible = "BASELINE", False
    prior = [r for r in log if r.get("kind") == "candidate" and "private_score" in r]
    rank = 1 + sum(1 for r in prior if r["private_score"] > score) if A.candidate else None
    rec = {"submission_id": f"priv_{len(log)+1:04d}", "label": label, "round": A.round, "kind": "candidate" if A.candidate else "baseline",
           "ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "seed": A.seed, "runtime_s": round(secs),
           "private_score": round(float(score), 4) if score == score else None, "rank": rank, "eligible_for_promotion": bool(eligible), "reason_code": reason}
    if not A.candidate: rec["_pinball"] = m.get("pinball")          # kept for the reference term only
    os.makedirs(os.path.join(LG, "registry"), exist_ok=True)
    with open(os.path.join(LG, "registry", "private_log.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}, indent=1))

if __name__ == "__main__": main()
