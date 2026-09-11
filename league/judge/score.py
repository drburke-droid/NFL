"""Deterministic visible judge for the NFL Model League.

Protocol (scoring config v1, frozen for Round 1):
  Evaluation window: seasons 2023-2024, weeks 1-18 (visible folds). Training rows: every visible
  row with t < test week (2016 onward), i.e. expanding weekly walk-forward. A candidate declares
  REFIT = "week" (fit before every test week) or "season" (fit once before each season; predict its
  weeks without within-season updates). Baselines use whichever applies to them.

Candidate contract (league/candidates/<exp_id>/model.py):
  REFIT: str                       "week" | "season"
  def fit_predict(train, test, seed) -> DataFrame indexed like `test` with columns
      pred (mean PPR), p10, p25, p50, p75, p90   (monotone; p50 may equal pred)
  No I/O of any kind inside model.py: the judge hands the frames in memory and rejects code that
  reads files, databases or the network (static check). Only columns present in the visible frame
  may be used. `actual_*` columns are present in `train` only.

Evaluation rows: players who actually played (actual_active == 1); a player projected but inactive is
not scored (the production generator zeroes known inactives from live lineups at T-90, which this
frame cannot see). Candidates still receive every row in `train`, with actual_active, and every row
in `test` must get a prediction.
Metrics on the evaluation window (fantasy PPR unless stated):
  mae, rmse, spearman (mean of weekly Spearman), pinball (mean over the 5 quantiles),
  cov80 (share of actuals inside [p10, p90]), per-season mae, coverage of predictions.
Composite (v1, normalised to the FROZEN FFA baseline where a ratio makes sense):
  0.45 * (ffa_mae / mae) + 0.20 * (ffa_rmse / rmse) + 0.15 * (spearman / ffa_spearman)
  + 0.10 * (ref_pinball / pinball) + 0.05 * max(0, 1 - 5|cov80 - 0.80|) + 0.05 * min_season(ffa_mae_s / mae_s)
  ref_pinball = the incumbent's (model_burke_prod: Model_Burke mean + production quantiles) pinball, so the distribution terms are comparable.
  Higher is better. The incumbent's composite is the number to beat.
Usage:
  python league/judge/score.py --candidate league/candidates/exp_001 [--seed 17] [--label name]
  python league/judge/score.py --predictions path.parquet --label ffa        (pre-made predictions)
"""
import os, re, sys, json, time, argparse, importlib.util, hashlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LG = os.path.join(ROOT, "league")
CONFIG = {"scoring_version": "v1", "eval_seasons": [2023, 2024], "weeks": list(range(1, 19)), "eval_active_only": True,
          "weights": {"mae": 0.45, "rmse": 0.20, "spearman": 0.15, "pinball": 0.10, "calibration": 0.05, "stability": 0.05},
          "min_coverage": 0.999, "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90]}
QS = CONFIG["quantiles"]
BANNED = re.compile(r"read_parquet|read_csv|read_sql|read_excel|read_json|sqlite3|open\(|urllib|requests\.|glob\.|os\.listdir|os\.walk|subprocess|pickle|joblib\.load|torch\.hub|np\.load")

def load_frame():
    v = pd.read_parquet(os.path.join(LG, "data", "visible_frame.parquet"))
    return v

def static_check(model_path):
    src = open(model_path, encoding="utf-8").read()
    hits = sorted(set(BANNED.findall(src)))
    return hits, hashlib.sha256(src.encode()).hexdigest()[:12]

def load_candidate(cdir):
    mp = os.path.join(cdir, "model.py")
    spec = importlib.util.spec_from_file_location("cand_model", mp); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def walk_forward(model, frame, seasons, seed, log=print):
    """Run the candidate over the evaluation weeks; returns predictions aligned to frame rows."""
    refit = getattr(model, "REFIT", "week")
    preds = []
    t0 = time.time()
    for s in seasons:
        weeks = sorted(frame[frame.season == s].week.unique())
        if refit == "season":
            train = frame[frame.season < s]
            test = frame[frame.season == s]
            out = model.fit_predict(train.copy(), test.drop(columns=[c for c in test.columns if c.startswith("actual_")]).copy(), seed)
            out = out.reindex(test.index); out["season"] = s; preds.append(out)
        else:
            for w in weeks:
                t = s * 100 + w
                train = frame[frame.t < t]; test = frame[frame.t == t]
                out = model.fit_predict(train.copy(), test.drop(columns=[c for c in test.columns if c.startswith("actual_")]).copy(), seed)
                out = out.reindex(test.index); preds.append(out)
        log(f"  season {s} done ({time.time()-t0:.0f}s)")
    p = pd.concat(preds)
    return p[["pred", "p10", "p25", "p50", "p75", "p90"]], time.time() - t0

def metrics(frame, p, ref_pinball=None, ffa=None):
    """frame: evaluation rows with actual_ppr; p: predictions aligned to frame.index."""
    d = frame.join(p, how="left")
    cov = float(d.pred.notna().mean())
    d = d.dropna(subset=["pred"])
    err = d.actual_ppr - d.pred
    out = {"n": int(len(d)), "coverage": round(cov, 4), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "bias": float(err.mean()), "spearman": float(d.groupby("t").apply(lambda x: spearmanr(x.pred, x.actual_ppr)[0]).mean())}
    if d[["p10", "p90"]].notna().all(axis=None):
        out["cov80"] = float(((d.actual_ppr >= d.p10) & (d.actual_ppr <= d.p90)).mean())
        out["pinball"] = float(np.mean([np.mean(np.maximum(q * (d.actual_ppr - d[f"p{int(q*100)}"]), (q - 1) * (d.actual_ppr - d[f"p{int(q*100)}"]))) for q in QS]))
        out["quantiles_monotone"] = bool((d.p10 <= d.p25).all() and (d.p25 <= d.p50).all() and (d.p50 <= d.p75).all() and (d.p75 <= d.p90).all())
    else:
        out["cov80"] = None; out["pinball"] = None; out["quantiles_monotone"] = None
    out["by_season"] = {str(int(s)): {"n": int(len(x)), "mae": float((x.actual_ppr - x.pred).abs().mean()), "rmse": float(np.sqrt(((x.actual_ppr - x.pred) ** 2).mean()))} for s, x in d.groupby("season")}
    out["by_position"] = {pos: {"n": int(len(x)), "mae": float((x.actual_ppr - x.pred).abs().mean()), "bias": float((x.actual_ppr - x.pred).mean())} for pos, x in d.groupby("position")}
    if ffa is not None:
        w = CONFIG["weights"]
        comp = w["mae"] * ffa["mae"] / out["mae"] + w["rmse"] * ffa["rmse"] / out["rmse"] + w["spearman"] * out["spearman"] / ffa["spearman"]
        comp += w["pinball"] * ((ref_pinball / out["pinball"]) if (out["pinball"] and ref_pinball) else 0.0)
        comp += w["calibration"] * (max(0.0, 1 - 5 * abs(out["cov80"] - 0.80)) if out["cov80"] is not None else 0.0)
        comp += w["stability"] * min(ffa["by_season"][s]["mae"] / out["by_season"][s]["mae"] for s in out["by_season"] if s in ffa["by_season"])
        out["composite"] = float(comp)
    return out

def registry_append(rec):
    os.makedirs(os.path.join(LG, "registry"), exist_ok=True)
    with open(os.path.join(LG, "registry", "experiments.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")

def load_registry():
    p = os.path.join(LG, "registry", "experiments.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def reference_metrics():
    """Frozen FFA baseline metrics and the incumbent's pinball, from the registry (baselines run first)."""
    regs = load_registry()
    ffa = next((r["metrics"] for r in reversed(regs) if r.get("label") == "ffa" and r.get("kind") == "visible"), None)
    inc = next((r["metrics"] for r in reversed(regs) if r.get("label") == "model_burke_prod" and r.get("kind") == "visible"), None)
    return ffa, (inc or {}).get("pinball")

def eval_rows(frame, seasons=None):
    """Rows the judge scores: evaluation seasons/weeks, and (v1) only players who actually played
    (actual_active == 1). Pre-kick inactives are handled live by the production generator with
    information this frame does not hold, so scoring them here would reward the wrong thing."""
    ev = frame[frame.season.isin(seasons or CONFIG["eval_seasons"]) & frame.week.isin(CONFIG["weeks"])]
    if CONFIG.get("eval_active_only") and "actual_active" in ev.columns: ev = ev[ev.actual_active == 1]
    return ev

def score_predictions(p, label, kind="visible", extra=None, frame=None):
    frame = frame if frame is not None else load_frame()
    ev = eval_rows(frame)
    ffa, ref_pin = reference_metrics()
    m = metrics(ev, p, ref_pinball=ref_pin, ffa=ffa)
    ok = m["coverage"] >= CONFIG["min_coverage"] and (m["quantiles_monotone"] in (True, None))
    rec = {"ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "label": label, "kind": kind, "scoring_version": CONFIG["scoring_version"],
           "valid": bool(ok), "metrics": m, **(extra or {})}
    registry_append(rec)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate"); ap.add_argument("--predictions"); ap.add_argument("--label"); ap.add_argument("--seed", type=int, default=17)
    A = ap.parse_args()
    frame = load_frame()
    if A.candidate:
        cdir = A.candidate.rstrip("/\\"); label = A.label or os.path.basename(cdir)
        hits, h = static_check(os.path.join(cdir, "model.py"))
        if hits:
            rec = {"ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "label": label, "kind": "visible", "valid": False, "reason": f"banned I/O in model.py: {hits}", "code_hash": h}
            registry_append(rec); print(json.dumps(rec, indent=1)); sys.exit(1)
        model = load_candidate(cdir)
        p, secs = walk_forward(model, frame, CONFIG["eval_seasons"], A.seed)
        os.makedirs(os.path.join(LG, "registry", "submissions"), exist_ok=True)
        p.to_parquet(os.path.join(LG, "registry", "submissions", f"{label}_visible.parquet"))
        rec = score_predictions(p, label, extra={"code_hash": h, "seed": A.seed, "runtime_s": round(secs), "refit": getattr(model, "REFIT", "week"), "candidate_dir": os.path.relpath(cdir, ROOT)}, frame=frame)
    else:
        p = pd.read_parquet(A.predictions); label = A.label or os.path.basename(A.predictions)
        rec = score_predictions(p, label, frame=frame)
    m = rec["metrics"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ("by_season", "by_position")}, indent=1))
    print("by season:", {s: round(x["mae"], 3) for s, x in m["by_season"].items()}, "| valid:", rec["valid"], "| composite:", round(m.get("composite", float("nan")), 4))

if __name__ == "__main__": main()
