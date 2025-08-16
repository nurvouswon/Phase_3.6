# App_learn_ranker.py
# =============================================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet (with robust join + fuzzy fallback)
# - Upload merged_leaderboards.csv + event-level parquet/csv (with hr_outcome)
# - Identity join order:
#   (0) game_date + batter_id  → (1) game_date + team_code + name_key
#   → (2) game_date + name_key → (3) game_date + player_name_norm
#   → (4) FUZZY within-day name_key (prefers same team), then join.
# - Trains ensemble ranker (LGB + XGB + CatBoost) and preserves prob_2tb/prob_rbi.
# - Exports labeled_leaderboard.csv + learning_ranker.pkl (+ optional name_map.csv if fuzzy used).
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import re
from datetime import datetime

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from unidecode import unidecode
from rapidfuzz import process, fuzz   # <-- fuzzy fallback

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# -------------------- Utilities --------------------
@st.cache_data(show_spinner=False)
def safe_read(fobj):
    name = str(getattr(fobj, "name", fobj)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(fobj)
    try:
        return pd.read_csv(fobj, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(fobj, encoding="latin1", low_memory=False)

def to_date_ymd(s, season_year=None):
    if pd.isna(s): return pd.NaT
    ss = str(s).strip()
    try:
        return pd.to_datetime(ss, errors="raise").normalize()
    except Exception:
        pass
    m = re.match(r"^\s*(\d{1,2})[^\d]+(\d{1,2})\s*$", ss)
    if m and season_year:
        mm = int(m.group(1)); dd = int(m.group(2))
        try:
            return pd.to_datetime(f"{int(season_year):04d}-{mm:02d}-{dd:02d}").normalize()
        except Exception:
            return pd.NaT
    return pd.NaT

def std_team(s):
    if pd.isna(s): return ""
    return str(s).upper().strip()

def clean_name_basic(s):
    if pd.isna(s): return ""
    s = unidecode(str(s)).upper().strip()
    s = re.sub(r"[\.\-']", " ", s)
    s = re.sub(r"\s+", " ", s)
    for suf in [", JR", " JR", " JR.", ", SR", " SR", " SR.", " II", " III", " IV"]:
        if s.endswith(suf): s = s[: -len(suf)]
    return s.strip()

def _strip_suffixes(s):
    s = s.upper()
    for suf in [", JR", " JR", " JR.", ", SR", " SR", " SR.", " II", " III", " IV"]:
        if s.endswith(suf): s = s[: -len(suf)]
    return s

def _squeeze_particles(last):
    if not last: return last
    last = last.replace(" MC ", " MC")
    parts = last.split()
    bad = {"DE","LA","DEL","DA","DI","DU","VAN","VON","DER","DEN"}
    packed = "".join([p for p in parts if p not in bad]) if len(parts) > 1 else last
    return packed

def make_name_key(raw_name: str) -> str:
    if pd.isna(raw_name): return ""
    s = unidecode(str(raw_name)).upper().strip()
    s = _strip_suffixes(s)
    toks = [t for t in s.replace(".", " ").replace("-", " ").split() if t]
    if not toks: return ""
    first = toks[0]
    last  = toks[-1] if len(toks) > 1 else ""
    last  = _squeeze_particles(last)
    if len(toks) >= 3:
        combo = _squeeze_particles(" ".join(toks[1:]))
        if len(combo) > len(last) + 2:
            last = combo
    return (" ".join([first[:1], last])).strip()

def groups_from_days(day_series: pd.Series):
    d = pd.to_datetime(day_series).dt.floor("D")
    return d.groupby(d.values).size().tolist()

def extract_batter_id(df):
    for cand in ["batter_id", "batter"]:
        if cand in df.columns:
            try:
                return df[cand].astype("Int64").astype(str).fillna("")
            except Exception:
                return df[cand].astype(str).fillna("")
    return pd.Series([""] * len(df), index=df.index)

# -------------------- UI --------------------
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level PARQUET/CSV (with hr_outcome)", type=["parquet", "csv"])
season_year = st.number_input(
    "Season year (only used if leaderboard 'game_date' is like '8_13')",
    min_value=2015, max_value=2100, value=2025, step=1
)

if not lb_file or not ev_file:
    st.info("Upload both files to continue.")
    st.stop()

# -------------------- Load --------------------
with st.spinner("Reading files..."):
    lb = safe_read(lb_file)
    ev = safe_read(ev_file)

st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

# -------------------- Normalize identity keys --------------------
req_lb = ["game_date", "player_name"]
req_ev = ["game_date", "player_name", "hr_outcome"]
for c in req_lb:
    if c not in lb.columns:
        st.error(f"Leaderboard missing required column: {c}")
        st.stop()
for c in req_ev:
    if c not in ev.columns:
        st.error(f"Event file missing required column: {c}")
        st.stop()

lb = lb.copy()
lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, season_year))
ev = ev.copy()
ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce").dt.normalize()

lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name_basic)
ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name_basic)

lb["team_code_std"] = (lb["team_code"].astype(str).apply(std_team) if "team_code" in lb.columns
                       else pd.Series([""] * len(lb)))
ev["team_code_std"] = (ev["team_code"].astype(str).apply(std_team) if "team_code" in ev.columns
                       else pd.Series([""] * len(ev)))

lb["name_key"] = lb["player_name"].astype(str).apply(make_name_key)
ev["name_key"] = ev["player_name"].astype(str).apply(make_name_key)

lb["batter_id_join"] = extract_batter_id(lb)
ev["batter_id_join"] = extract_batter_id(ev)

# -------------------- Quick diagnostics BEFORE join --------------------
lb_dates = pd.to_datetime(lb["game_date"]).dt.date.unique()
ev_dates = pd.to_datetime(ev["game_date"]).dt.date.unique()
date_overlap = sorted(set(lb_dates).intersection(set(ev_dates)))
st.write(f"🔍 Date overlap count: {len(date_overlap)}")
if len(date_overlap) == 0:
    st.error("❌ No date overlap between leaderboard and event parquet. Check the actual dates in each file.")
    st.stop()

st.write("Leaderboard date range:", str(pd.to_datetime(lb["game_date"]).min().date()), "→", str(pd.to_datetime(lb["game_date"]).max().date()))
st.write("Event date range:", str(pd.to_datetime(ev["game_date"]).min().date()), "→", str(pd.to_datetime(ev["game_date"]).max().date()))

# -------------------- Build per-day labels from event --------------------
ev_daily = (
    ev.groupby(
        ["game_date", "player_name_norm", "name_key", "team_code_std", "batter_id_join"],
        dropna=False
    )["hr_outcome"]
    .max()
    .reset_index()
)

# -------------------- Deterministic join attempts --------------------
def do_merge(l, r, on, tag):
    cols = list(dict.fromkeys(on + ["hr_outcome"]))
    m = l.merge(r[cols], on=on, how="left", suffixes=("", "_y"))
    return m, tag

def sequential_join(lb0, ev0):
    l = lb0.copy()
    r = ev0.copy()

    # (0) game_date + batter_id_join (only rows where both non-empty)
    has_l = l["batter_id_join"].str.len().gt(0).any()
    has_r = r["batter_id_join"].str.len().gt(0).any()
    if has_l and has_r:
        l0 = l[l["batter_id_join"].str.len().gt(0)].copy()
        r0 = r[r["batter_id_join"].str.len().gt(0)].copy()
        m0, t0 = do_merge(l0, r0, ["game_date", "batter_id_join"], "game_date + batter_id")
        l = pd.concat([m0, l[~l.index.isin(m0.index)]], ignore_index=True)
        if l["hr_outcome"].notna().any():
            return l, t0, False

    # (1) game_date + team_code_std + name_key
    if ("team_code_std" in l.columns) and ("team_code_std" in r.columns):
        m1, t1 = do_merge(l, r, ["game_date", "team_code_std", "name_key"], "game_date + team_code + name_key")
        if m1["hr_outcome"].notna().any():
            return m1, t1, False

    # (2) game_date + name_key
    m2, t2 = do_merge(l, r, ["game_date", "name_key"], "game_date + name_key")
    if m2["hr_outcome"].notna().any():
        return m2, t2, False

    # (3) game_date + player_name_norm
    m3, t3 = do_merge(l, r, ["game_date", "player_name_norm"], "game_date + player_name_norm")
    return m3, t3, True  # True → needs fuzzy
# -----------------------------------------------------------------------

with st.spinner("Joining labels (deterministic passes)..."):
    merged, join_tag, needs_fuzzy = sequential_join(lb, ev_daily)

labeled = merged[merged["hr_outcome"].notna()].copy()
st.write(f"🔎 Deterministic join tag: **{join_tag}** | Labeled so far: {len(labeled)} / {len(merged)}")

# -------------------- Fuzzy fallback (within-day) --------------------
name_map_rows = []
if (len(labeled) == 0) or needs_fuzzy:
    st.warning("Running fuzzy name resolver within each date (prefers same team when available)...")
    m = merged.copy()

    # Work only on rows still unlabeled
    mask_un = m["hr_outcome"].isna()
    if mask_un.any():
        # Build per-day lookup for event side
        ev_by_day = {d: df.copy() for d, df in ev_daily.groupby(ev_daily["game_date"].dt.floor("D"))}

        # Iterate unlabeled rows and attempt a fuzzy map
        for idx in m[mask_un].index:
            row = m.loc[idx]
            d = pd.to_datetime(row["game_date"]).floor("D")
            if d not in ev_by_day:
                continue
            evd = ev_by_day[d]

            # Prefer same team pool if available
            if "team_code_std" in evd.columns and pd.notna(row.get("team_code_std", "")) and str(row.get("team_code_std","")) != "":
                pool = evd[evd["team_code_std"] == str(row["team_code_std"])].copy()
                if pool.empty:
                    pool = evd.copy()
            else:
                pool = evd.copy()

            cand_keys = pool["name_key"].astype(str).tolist()
            target = str(row["name_key"])

            # If name_key is empty, try player_name_norm
            if not target:
                target = str(row["player_name_norm"])

            if not cand_keys or not target:
                continue

            # Strict then looser thresholds
            match = process.extractOne(
                target, cand_keys,
                scorer=fuzz.WRatio
            )
            if match and match[1] >= 92:
                best = match[0]
            else:
                # try a slightly looser threshold
                if match and match[1] >= 88:
                    best = match[0]
                else:
                    continue

            ev_row = pool.loc[pool["name_key"] == best]
            if not ev_row.empty:
                hr_val = float(ev_row.iloc[0]["hr_outcome"])
                m.loc[idx, "hr_outcome"] = hr_val
                name_map_rows.append({
                    "game_date": str(d.date()),
                    "lb_player": row["player_name"],
                    "lb_name_key": row["name_key"],
                    "ev_player": ev_row.iloc[0]["player_name_norm"],
                    "ev_name_key": ev_row.iloc[0]["name_key"],
                    "team_lb": row.get("team_code_std",""),
                    "team_ev": ev_row.iloc[0]["team_code_std"],
                    "score": match[1]
                })

    merged = m
    labeled = merged[merged["hr_outcome"].notna()].copy()
    st.write(f"🧩 After fuzzy map, labeled rows: {len(labeled)} / {len(merged)}")
    if name_map_rows:
        nm = pd.DataFrame(name_map_rows)
        nm_csv = io.StringIO()
        nm.to_csv(nm_csv, index=False)
        st.download_button("⬇️ Download name_map (fuzzy matches) CSV", nm_csv.getvalue(), "name_map.csv", "text/csv")

if len(labeled) == 0:
    # Hard-stop with concrete pointers
    # Show example of first 10 names per overlapping date to help user verify
    st.error(
        "❌ No label matches found after deterministic + fuzzy join.\n\n"
        "Quick checks:\n"
        "• Confirm your event parquet **actually contains the same dates** as the leaderboard (the app printed both ranges above).\n"
        "• Ensure leaderboard 'player_name' strings match your prediction app’s output (accents/suffixes are normalized here).\n"
        "• If you can ever export 'batter_id' in the leaderboard, matching becomes near-perfect."
    )
    st.stop()

# -------------------- Feature set --------------------
candidate_feats = [
    "ranked_probability",
    "hr_probability_iso_T",
    "final_multiplier",
    "overlay_multiplier",
    "weak_pitcher_factor",
    "hot_streak_factor",
    "rrf_aux",
    "model_disagreement",
    "prob_2tb",
    "prob_rbi",
    "final_multiplier_raw",
    "temp", "humidity", "wind_mph",
]
avail = [c for c in candidate_feats if c in labeled.columns]
if not avail:
    st.error("No usable features found in leaderboard; need at least the core columns.")
    st.stop()

X = labeled[avail].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
y = labeled["hr_outcome"].astype(int).values
groups = groups_from_days(labeled["game_date"])

if X.shape[0] == 0 or X.shape[1] == 0:
    st.error("Empty feature matrix after filtering. Double-check your leaderboard columns.")
    st.stop()

# -------------------- Train ranker ensemble --------------------
st.subheader("Training day-wise ranker ensemble")
st.write(f"Features used ({len(avail)}): {', '.join(avail)}")

rk_lgb = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=700, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    random_state=42
)
rk_lgb.fit(X, y, group=groups)
pred_lgb = rk_lgb.predict(X)

pred_xgb = None; rk_xgb = None
try:
    rk_xgb = xgb.XGBRanker(
        n_estimators=700, learning_rate=0.06, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        objective="rank:pairwise", random_state=42, tree_method="hist"
    )
    rk_xgb.fit(X, y, group=groups, verbose=False)
    pred_xgb = rk_xgb.predict(X)
except Exception as e:
    st.warning(f"XGBRanker not used: {e}")

pred_cb = None; rk_cb = None
try:
    rk_cb = cb.CatBoost(
        iterations=1200, learning_rate=0.05, depth=7,
        loss_function="YetiRank", random_seed=42, verbose=False
    )
    rk_cb.fit(X, y, group_id=np.concatenate([[i]*g for i, g in enumerate(groups)]))
    pred_cb = rk_cb.predict(X).flatten()
except Exception as e:
    st.warning(f"CatBoost YetiRank not used: {e}")

preds = [p for p in [pred_lgb, pred_xgb, pred_cb] if p is not None]
ens_train = np.mean(np.column_stack(preds), axis=1) if len(preds) > 1 else pred_lgb

# Per-day ndcg (sanity)
try:
    ndcgs = []
    for day, df_day in labeled.groupby(labeled["game_date"].dt.floor("D")):
        idx = df_day.index
        y_true = df_day["hr_outcome"].values.reshape(1, -1)
        y_score = ens_train[idx].reshape(1, -1)
        nd = ndcg_score(y_true, y_score, k=min(10, y_true.shape[1]))
        ndcgs.append(float(nd))
    st.write(f"NDCG@10 (mean across days): {np.mean(ndcgs):.4f}")
except Exception:
    pass

st.success("✅ Ranker trained.")

# -------------------- Save artifacts --------------------
labeled_out = labeled.copy()
labeled_out = labeled_out.sort_values(["game_date", "ranked_probability"], ascending=[True, False])
csv_buf = io.StringIO()
labeled_out.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Download Labeled Leaderboard CSV",
    data=csv_buf.getvalue(),
    file_name="labeled_leaderboard.csv",
    mime="text/csv"
)

bundle = {
    "features": avail,
    "model_type": "ranker_ensemble",
    "models": {
        "lgb": rk_lgb,
        "xgb": rk_xgb,
        "cat": rk_cb,
    },
    "join_info": {
        "deterministic_strategy": "0→1→2→3 (id/team/name fallbacks)",
        "used_fuzzy": bool(len(labeled) and labeled["hr_outcome"].isna().sum() < len(merged)),
        "labeled_rows": int(len(labeled)),
        "total_rows": int(len(merged)),
    },
}
pkl_bytes = io.BytesIO()
pickle.dump(bundle, pkl_bytes)
pkl_bytes.seek(0)
st.download_button(
    "⬇️ Download learning_ranker.pkl",
    data=pkl_bytes,
    file_name="learning_ranker.pkl",
    mime="application/octet-stream"
)

st.caption("All 3 rankers tried; 2+TB & RBI features included if present. Robust label join with fuzzy fallback.")
