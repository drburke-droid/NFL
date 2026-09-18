"""fetch_schedule_lines must not lose games a later nflverse pull no longer carries.

nflverse's spread_line / total_line for the current season is a live rolling field:
a game carries a line only while a book has it on the board. The ingestion drops
unpriced games, so replacing the table wholesale discarded every current-season game
the newest pull happened to miss — which is how the 2026 table came to hold 78 of 272
games, with no DET @ BUF in week 2.
"""
import importlib.util
import os
import sqlite3
import sys
import types

import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCHED = {"df": None}


def _load_module():
    """Import the script with nflreadpy stubbed out (it is not a test dependency)."""
    class _Pl:
        def to_pandas(self):
            return _SCHED["df"].copy()

    stub = types.ModuleType("nflreadpy")
    stub.load_schedules = lambda: _Pl()
    sys.modules["nflreadpy"] = stub
    spec = importlib.util.spec_from_file_location(
        "fetch_schedule_lines", os.path.join(REPO, "scripts", "fetch_schedule_lines.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fsl = _load_module()


def schedule(rows):
    """rows: (season, week, game_type, away, home, spread_line, total_line)"""
    return pd.DataFrame([{"season": s, "week": w, "game_type": gt, "away_team": a, "home_team": h,
                          "spread_line": sp, "total_line": to,
                          "home_moneyline": -150.0, "away_moneyline": 130.0}
                         for s, w, gt, a, h, sp, to in rows])


# what the table froze with: week 1 plus look-ahead lines out to week 6, no week-2 DET @ BUF
FROZEN = schedule([
    (2026, 1, "REG", "NO", "DET", -7.0, 48.5),
    (2026, 1, "REG", "BUF", "HOU", 1.5, 45.5),
    (2026, 4, "REG", "DET", "CAR", 3.0, 47.5),
    (2026, 6, "REG", "BUF", "LV", -6.5, 47.5),
    (2025, 1, "REG", "AAA", "BBB", 2.0, 44.0),
])
# a pull taken later: the look-aheads have come off the board, week 2 is finally priced,
# and the week-1 line has moved
LIVE = schedule([
    (2026, 1, "REG", "NO", "DET", -9.0, 50.5),
    (2026, 1, "REG", "BUF", "HOU", 1.5, 45.5),
    (2026, 2, "REG", "DET", "BUF", 5.5, 54.5),
    (2026, 3, "REG", "LAC", "BUF", -2.5, 48.5),
    (2025, 1, "REG", "AAA", "BBB", 2.0, 44.0),
])


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs the script against a throwaway DB and an isolated export directory."""
    db = tmp_path / "odds.db"
    con = sqlite3.connect(db)
    fsl.build(FROZEN).to_sql(fsl.TABLE, con, if_exists="replace", index=False)
    pd.DataFrame({"season": [2026], "week": [1], "team": ["DET"], "season_type": ["REG"]}
                 ).to_sql("nflv_weekly", con, if_exists="replace", index=False)
    con.close()
    os.makedirs(tmp_path / "data" / "sabersim")
    monkeypatch.setattr(fsl, "DB", str(db))
    monkeypatch.setattr(fsl, "ROOT", str(tmp_path))

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["fetch_schedule_lines.py", *argv])
        _SCHED["df"] = LIVE
        fsl.main()
        con = sqlite3.connect(db)
        out = pd.read_sql(f"SELECT * FROM {fsl.TABLE}", con)
        con.close()
        return out

    return _run


def teams(df, season, week):
    return sorted(set(df[(df.season == season) & (df.week == week)].team))


def test_upsert_keeps_games_the_new_pull_dropped(run):
    out = run()
    assert teams(out, 2026, 4) == ["CAR", "DET"]
    assert teams(out, 2026, 6) == ["BUF", "LV"]


def test_upsert_adds_games_the_old_table_lacked(run):
    out = run()
    assert teams(out, 2026, 2) == ["BUF", "DET"]
    assert teams(out, 2026, 3) == ["BUF", "LAC"]


def test_fresh_rows_win_on_conflict(run):
    det = run().query("season == 2026 and week == 1 and team == 'DET'").iloc[0]
    assert det.game_total == 50.5
    assert det.team_spread == 9.0          # spread_line -9 => home favoured by 9


def test_settled_history_untouched(run):
    assert teams(run(), 2025, 1) == ["AAA", "BBB"]


def test_no_duplicate_keys(run):
    out = run()
    assert out[fsl.KEY].apply(tuple, axis=1).is_unique
    assert len(out) == 14                  # 10 from the pull + 4 kept


def test_replace_flag_still_rebuilds_from_scratch(run):
    out = run("--replace")
    assert teams(out, 2026, 4) == []
    assert teams(out, 2026, 2) == ["BUF", "DET"]


def test_parquet_export_refuses_to_shrink(run, tmp_path):
    fp = tmp_path / "data" / "sabersim" / "game_lines_2023_2026.parquet"
    seed = fsl.build(FROZEN)[fsl.PQ_COLS]
    seed.to_parquet(fp, index=False)
    run("--replace", "--export-parquet")
    assert len(pd.read_parquet(fp)) == len(seed)


def test_parquet_export_writes_when_it_grows(run, tmp_path):
    fp = tmp_path / "data" / "sabersim" / "game_lines_2023_2026.parquet"
    run("--export-parquet")
    written = pd.read_parquet(fp)
    assert list(written.columns) == fsl.PQ_COLS
    assert sorted(set(written[written.week == 2].team)) == ["BUF", "DET"]
