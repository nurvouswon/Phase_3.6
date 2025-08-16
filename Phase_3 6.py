# App_learn_ranker.py
# =============================================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged_leaderboards.csv (across days) + event-level parquet/csv (with hr_outcome)
# - Robust label join (date + id → date+team+name_key → date+name_key → date+player_name_norm)
# - Trains day-wise ranker ensemble (LGBMRanker + XGBRanker + CatBoost YetiRank)
# - Keeps 2TB/RBI (prob_2tb/prob_rbi) as features if present (no plots)
# - Exports: labeled leaderboard CSV + learning_ranker.pkl (features + ensemble)
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
    """Accepts 'YYYY-MM-DD' or 'M_D' style (e.g., '8_13') if season_year provided."""
    if pd.isna(s):
        return pd.NaT
    ss = str(s).strip()
    # direct ISO
    try:
        return pd.to_datetime(ss, errors="raise").normalize()
    except Exception:
        pass
    # M_D fallback like 8_13 or 8-13 etc.
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
    # remove suffixes
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
    tokens = [t for t in s.replace(".", " ").replace("-", " ").split() if t]
    if not tokens: return ""
    first = tokens[0]
    last  = tokens[-1] if len(tokens) > 1 else ""
    last  = _squeeze_particles(last)
    # handle multi-token last names (DE LA CRUZ -> DELACRUZ)
    if len(tokens) >= 3:
        combo = _squeeze_particles(" ".join(tokens[1:]))
        if len(combo) > len(last) + 2:
            last = combo
    key = (first[:1] + " " + last).strip()
    return " ".join(key.split())

def groups_from_days(day_series: pd.Series):
    """Return LightGBM/XGB group sizes from a day series."""
    d = pd.to_datetime(day_series).dt.floor("D")
    return d.groupby(d.values).size().tolist()

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
# game_date
if "game_date" not in lb.columns:
    st.error("Leaderboard must contain 'game_date' column.")
    st.stop()

lb = lb.copy()
lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, season_year))
if lb["game_date"].isna().all():
    st.error("Could not parse any 'game_date' in leaderboard. Ensure YYYY-MM-DD or M_D with Season year.")
    st.stop()

ev = ev.copy()
if "game_date" not in ev.columns:
    st.error("Event file must contain 'game_date'.")
    st.stop()
ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce").dt.normalize()

# names
if "player_name" not in lb.columns:
    st.error("Leaderboard must have 'player_name'.")
    st.stop()
if "player_name" not in ev.columns:
    st.error("Event file must have 'player_name'.")
    st.stop()

lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name_basic)
ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name_basic)

# team
lb["team_code_std"] = (lb["team_code"].astype(str).apply(std_team) if "team_code" in lb.columns
                       else pd.Series([""] * len(lb)))
ev["team_code_std"] = (ev["team_code"].astype(str).apply(std_team) if "team_code" in ev.columns
                       else pd.Series([""] * len(ev)))

# name_key like predictor
lb["name_key"] = lb["player_name"].astype(str).apply(make_name_key)
ev["name_key"] = ev["player_name"].astype(str).apply(make_name_key)

# batter_id (string-typed for join)
def extract_batter_id(df):
    for cand in ["batter_id", "batter"]:
        if cand in df.columns:
            s = df[cand].astype("Int64", errors="ignore") if pd.api.types.is_integer_dtype(df[cand]) else df[cand]
            return s.astype(str).fillna("")
    return pd.Series([""] * len(df), index=df.index)

lb["batter_id_join"] = extract_batter_id(lb)
ev["batter_id_join"] = extract_batter_id(ev)

# -------------------- Build daily labels from event --------------------
if "hr_outcome" not in ev.columns:
    st.error("Event file must include 'hr_outcome' (0/1).")
    st.stop()

# Aggregate to per-game, per-batter label (any HR that day)
ev_daily = (
    ev.groupby(
        ["game_date", "player_name_norm", "name_key", "team_code_std", "batter_id_join"],
        dropna=False
    )["hr_outcome"]
    .max()
    .reset_index()
)

# -------------------- Robust join --------------------
def safe_join(left, right):
    left = left.copy()
    right = right.copy()

    def do_merge(l, r, on, tag):
        cols = list(dict.fromkeys(on + ["hr_outcome"]))
        m = l.merge(r[cols], on=on, how="left", suffixes=("", "_y"))
        return m, tag

    # Attempt 0: date + batter_id (if present on both sides non-empty)
    has_l = left["batter_id_join"].astype(str).str.len().gt(0).any() if "batter_id_join" in left.columns else False
    has_r = right["batter_id_join"].astype(str).str.len().gt(0).any() if "batter_id_join" in right.columns else False
    if has_l and has_r:
        l_id = left[left["batter_id_join"].astype(str).str.len().gt(0)].copy()
        r_id = right[right["batter_id_join"].astype(str).str.len().gt(0)].copy()
        l_id["batter_id_join"] = l_id["batter_id_join"].astype(str)
        r_id["batter_id_join"] = r_id["batter_id_join"].astype(str)
        m0, tag0 = do_merge(l_id, r_id, ["game_date", "batter_id_join"], "game_date + batter_id")
        # combine with leftovers to keep row count
        leftover = left[~left.index.isin(m0.index)]
        merged = pd.concat([m0, leftover], ignore_index=True)
        if merged["hr_outcome"].notna().any():
            return merged, tag0

    # Attempt 1: date + team + name_key
    if ("team_code_std" in left.columns) and ("team_code_std" in right.columns):
        m1, tag1 = do_merge(left, right, ["game_date", "team_code_std", "name_key"], "game_date + team_code + name_key")
        if m1["hr_outcome"].notna().any():
            return m1, tag1

    # Attempt 2: date + name_key
    m2, tag2 = do_merge(left, right, ["game_date", "name_key"], "game_date + name_key")
    if m2["hr_outcome"].notna().any():
        return m2, tag2

    # Attempt 3: date + player_name_norm
    m3, tag3 = do_merge(left, right, ["game_date", "player_name_norm"], "game_date + player_name_norm")
    return m3, tag3

with st.spinner("Joining labels..."):
    merged, join_tag = safe_join(lb, ev_daily)

labeled = merged[merged["hr_outcome"].notna()].copy()
st.write(f"🔎 Join strategy used: **{join_tag}**")
st.write(f"✅ Labeled rows: {len(labeled):,} (out of {len(merged):,})")

if len(labeled) == 0:
    st.error(
        "❌ No label matches found after join.\n\n"
        "Quick checks you can do in the CSV:\n"
        "• Ensure 'game_date' is YYYY-MM-DD (or M_D with the Season year above).\n"
        "• Keep 'player_name' as exported by your prediction app (accents/suffixes are normalized here).\n"
        "• Confirm your event parquet covers the same dates.\n"
        "• (Optional) If you ever include 'batter_id' in the leaderboard, matching becomes near-perfect."
    )
    st.stop()

# -------------------- Feature set --------------------
# Keep everything you expect from the leaderboard; use whatever exists.
candidate_feats = [
    # core from your leaderboard export
    "ranked_probability",
    "hr_probability_iso_T",
    "final_multiplier",
    "overlay_multiplier",
    "weak_pitcher_factor",
    "hot_streak_factor",
    "rrf_aux",
    "model_disagreement",
    # extras you asked to keep
    "prob_2tb",
    "prob_rbi",
    "final_multiplier_raw",
    # weather/context (treated as numeric if already numeric)
    "temp", "humidity", "wind_mph",
]

# filter to available numeric columns
avail = [c for c in candidate_feats if c in labeled.columns]
if not avail:
    st.error("No usable features found in leaderboard. Make sure it includes the core columns (ranked_probability, hr_probability_iso_T, final_multiplier, etc.).")
    st.stop()

X = labeled[avail].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
y = labeled["hr_outcome"].astype(int).values
groups = groups_from_days(labeled["game_date"])

if X.shape[1] == 0 or X.shape[0] == 0:
    st.error("Empty feature matrix after filtering. Double-check your leaderboard columns.")
    st.stop()

# -------------------- Train ranker ensemble --------------------
st.subheader("Training day-wise ranker ensemble")
st.write(f"Features used ({len(avail)}): {', '.join(avail)}")

# LightGBM Ranker
rk_lgb = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=700, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    random_state=42
)
rk_lgb.fit(X, y, group=groups)

pred_lgb = rk_lgb.predict(X)

# Try XGBoost ranker (optional)
pred_xgb = None
rk_xgb = None
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

# Try CatBoost YetiRank (optional)
pred_cb = None
rk_cb = None
try:
    rk_cb = cb.CatBoost(
        iterations=1200, learning_rate=0.05, depth=7,
        loss_function="YetiRank", random_seed=42, verbose=False
    )
    rk_cb.fit(X, y, group_id=np.concatenate([[i]*g for i, g in enumerate(groups)]))
    pred_cb = rk_cb.predict(X).flatten()
except Exception as e:
    st.warning(f"CatBoost YetiRank not used: {e}")

# Blend rankers (simple average of available)
preds = [p for p in [pred_lgb, pred_xgb, pred_cb] if p is not None]
ens_train = np.mean(np.column_stack(preds), axis=1) if len(preds) > 1 else pred_lgb

# Report NDCG by day (sanity)
try:
    # Build per-day ndcg@10
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
# Save labeled leaderboard for your QA
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

# Save model bundle your predictor can load (it reads 'features' + 'model')
bundle = {
    "features": avail,
    "model_type": "ranker_ensemble",
    "models": {
        "lgb": rk_lgb,
        "xgb": rk_xgb,
        "cat": rk_cb,
    },
    "join_info": {
        "strategy": join_tag,
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

st.caption("This bundle contains the ranker ensemble + the exact feature list used. Your predictor can load and apply it to reorder the day.")
